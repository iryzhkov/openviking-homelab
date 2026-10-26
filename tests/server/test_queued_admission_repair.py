# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Queued resource work keeps its original handoff and account lifetime.

P1: a persisted AddResource SOURCE job whose original handoff was lost in the
same account lifetime must fail closed instead of reacquiring a fresh owner,
and must not delete another writer's empty target.

P3: resource parent freshness runs after the child scope closes. It must
recheck the original account identity under final native coverage before
touching parent sidecars; unbound resource content-write refresh shapes must
refuse at consumer admission.

The service, native filesystem, leases, FileStore registry, queues and
handlers are real; models are synthetic. Wrappers only hold a queue message
or pause at the named boundary and always call through.
"""

import json
from types import SimpleNamespace

import pytest

from openviking.storage.queuefs.add_resource_processor import AddResourceProcessor
from openviking.storage.queuefs.named_queue import NamedQueue
from openviking.storage.queuefs.process_result import ProcessOutcome
from openviking.storage.queuefs.resource_embedding_fence import ResourceEmbeddingConflict
from tests.server.conftest import sample_markdown_file as sample_markdown_file
from tests.server.conftest import service as service
from tests.server.conftest import temp_dir as temp_dir
from tests.server.test_real_resource_producer_e2e import (
    _recreate_account,
    _root_ctx,
)
from tests.server.test_real_resource_producer_e2e import admitted_service as admitted_service

TARGET = "viking://resources/held"


async def _hold_add_resource(service, sample_markdown_file, monkeypatch):
    """Run the real request path; keep its persisted AddResource message."""
    held = []
    original = NamedQueue.enqueue

    async def hold(queue, data):
        if getattr(queue, "name", "") == "AddResource" and not held:
            held.append(json.loads(json.dumps(data)))
            return "held"
        return await original(queue, data)

    monkeypatch.setattr(NamedQueue, "enqueue", hold)
    result = await service.resources.add_resource(
        path=str(sample_markdown_file), ctx=_root_ctx(), to=TARGET, reason="p1", wait=False
    )
    monkeypatch.setattr(NamedQueue, "enqueue", original)
    assert held, result
    payload = held[0].get("data", held[0])
    if isinstance(payload, str):
        payload = json.loads(payload)
    assert payload.get("lock_handoff") and payload.get("resource_creation_identity"), payload
    return payload


def _processor(service):
    return AddResourceProcessor(service.resources, "AddResource", service.viking_fs)


async def test_lost_original_handoff_fails_closed_and_keeps_foreign_empty_target(
    admitted_service, sample_markdown_file, monkeypatch
):
    service = admitted_service
    fs = service.viking_fs
    agfs = fs._async_agfs
    payload = await _hold_add_resource(service, sample_markdown_file, monkeypatch)
    # Same account lifetime: the original owner settles, then another writer
    # leaves an empty directory at the reserved target.
    adopted = await agfs.pathlock_adopt(payload["lock_handoff"])
    await agfs.pathlock_release(adopted)
    target_path = fs._uri_to_path(TARGET, ctx=_root_ctx())
    if not await fs.exists(TARGET, ctx=_root_ctx()):
        foreign = await agfs.pathlock_acquire_tree(target_path.rsplit("/", 1)[0])
        try:
            await agfs.mkdir(target_path)
        finally:
            await agfs.pathlock_release(foreign)
    reacquired = []
    original_reacquire = service.resources.reacquire_add_resource_job_lock

    async def record_reacquire(*args, **kwargs):
        reacquired.append(args)
        return await original_reacquire(*args, **kwargs)

    monkeypatch.setattr(service.resources, "reacquire_add_resource_job_lock", record_reacquire)
    payload["lock_handoff_retry"] = 2  # no requeue: observe the terminal refusal
    result = await _processor(service).on_dequeue({"data": json.dumps(payload)})
    assert result.outcome is ProcessOutcome.FAILED, result
    assert "lock_handoff" in (result.error or ""), result
    assert reacquired == []
    # The foreign empty directory survives and receives no content.
    assert await fs.exists(TARGET, ctx=_root_ctx())
    entries = await fs.ls(TARGET, ctx=_root_ctx(), show_all_hidden=True)
    assert [e for e in entries if e.get("name") not in {".", "..", ".path.ovlock"}] == []
    lease = await agfs.pathlock_acquire_tree(target_path)
    await agfs.pathlock_release(lease)


async def test_missing_resource_handoff_fails_closed(admitted_service):
    msg = SimpleNamespace(lock_handoff=None, resource_creation_identity="original", root_uri=TARGET)
    with pytest.raises(ValueError, match="original lock handoff"):
        await _processor(admitted_service)._load_lock(msg, _root_ctx())


async def test_nonresource_job_keeps_reacquire_recovery(admitted_service, monkeypatch):
    service = admitted_service
    calls = []

    async def reacquire(root_uri, ctx):
        calls.append(root_uri)
        return {"lease_ref": "recovered"}

    monkeypatch.setattr(service.resources, "reacquire_add_resource_job_lock", reacquire)
    msg = SimpleNamespace(
        lock_handoff={"lease_ref": "gone"},
        resource_creation_identity=None,
        root_uri="viking://user/default/memories/x",
    )
    assert await _processor(service)._load_lock(msg, _root_ctx()) == {"lease_ref": "recovered"}
    assert calls == ["viking://user/default/memories/x"]


async def test_live_original_handoff_ingests(admitted_service, sample_markdown_file, monkeypatch):
    service = admitted_service
    payload = await _hold_add_resource(service, sample_markdown_file, monkeypatch)
    result = await _processor(service).on_dequeue({"data": json.dumps(payload)})
    assert result.outcome is ProcessOutcome.SUCCESS, result
    assert await service.viking_fs.exists(TARGET + "/sample.md", ctx=_root_ctx())


@pytest.mark.parametrize("kind", ["batch", "abstract_overview", "memory"])
def test_unbound_resource_refresh_shapes_refuse_at_admission(kind):
    from openviking.service.content_write_admission import content_write_admission

    msg = SimpleNamespace(resource_id=TARGET, refresh={"kind": kind}, lock_handoff={"x": 1})
    with pytest.raises(ResourceEmbeddingConflict, match="Unbound resource"):
        content_write_admission(msg)


def test_admission_keeps_nonresource_and_noop_refresh():
    from openviking.service.content_write_admission import content_write_admission

    other = SimpleNamespace(
        resource_id="viking://user/default/memories/x", refresh={"kind": "batch"}, lock_handoff=None
    )
    noop = SimpleNamespace(resource_id=TARGET, refresh={"kind": "none"}, lock_handoff=None)
    assert content_write_admission(other) is None
    assert content_write_admission(noop) is None


async def _seed_root_sidecars(fs, label):
    seeded = {}
    for name in (".abstract.md", ".overview.md"):
        path = fs._uri_to_path("viking://resources/" + name, ctx=_root_ctx())
        body = (
            "---\ndirectory: viking://resources/\ngenerated_by:\n"
            "  component: SemanticProcessor\n  trigger: parent_refresh\n"
            "freshness:\n  total_entries: 100\n  sampled_entries: 100\n"
            "  unsampled_entries: 0\n  pending_child_changes: 0\n---\n\n"
            f"{label} lifetime body\n"
        ).encode()
        await fs._async_agfs.write(path, body)
        seeded[name] = body
    return seeded


async def _read_root_sidecars(fs):
    out = {}
    for name in (".abstract.md", ".overview.md"):
        path = fs._uri_to_path("viking://resources/" + name, ctx=_root_ctx())
        out[name] = await fs._async_agfs.cat(path)
    return out


@pytest.mark.parametrize("lifetime", ["recreated", "same"])
async def test_parent_freshness_rechecks_original_lifetime_after_child_scope(
    admitted_service, sample_markdown_file, monkeypatch, lifetime
):
    from openviking.storage.queuefs.semantic_processor import SemanticProcessor

    service = admitted_service
    fs = service.viking_fs
    original_parent = SemanticProcessor._enqueue_parent_refresh
    original_fresh = SemanticProcessor._resource_parent_freshness
    observed = []
    decisions = []

    async def record_freshness(self, *args, **kwargs):
        decision = await original_fresh(self, *args, **kwargs)
        decisions.append(decision)
        return decision

    monkeypatch.setattr(SemanticProcessor, "_resource_parent_freshness", record_freshness)

    async def pause_after_child_scope(self, msg, uri, *, l0_body_changed):
        if msg.context_type != "resource" or observed:
            return await original_parent(self, msg, uri, l0_body_changed=l0_body_changed)
        # Child scope is closed here; seed parent sidecars of the lifetime
        # that will be current when parent freshness runs.
        if lifetime == "recreated":
            await _recreate_account(service)
        seeded = await _seed_root_sidecars(fs, lifetime)
        queued = []
        original_enqueue = NamedQueue.enqueue

        async def record(queue, data):
            if getattr(queue, "name", "") == "Semantic":
                queued.append(data)
            return await original_enqueue(queue, data)

        monkeypatch.setattr(NamedQueue, "enqueue", record)
        try:
            await original_parent(self, msg, uri, l0_body_changed=l0_body_changed)
        finally:
            monkeypatch.setattr(NamedQueue, "enqueue", original_enqueue)
        observed.append((seeded, await _read_root_sidecars(fs), queued))

    monkeypatch.setattr(SemanticProcessor, "_enqueue_parent_refresh", pause_after_child_scope)
    await service.resources.add_resource(
        path=str(sample_markdown_file), ctx=_root_ctx(), reason="p3", wait=True
    )
    assert observed, "resource parent freshness never ran"
    seeded, after, queued = observed[0]
    assert decisions, "resource parent freshness was not reached"
    if lifetime == "recreated":
        # Refused under final coverage: no sidecar write, no parent work.
        assert decisions[0] is None
        assert after == seeded
        assert queued == []
    else:
        # Same lifetime: real freshness runs on structured sidecars and
        # records the pending child change.
        assert decisions[0] is not None
        assert after != seeded
        assert b"pending_child_changes: 1" in after[".abstract.md"]
        assert b"pending_child_changes: 1" in after[".overview.md"]
