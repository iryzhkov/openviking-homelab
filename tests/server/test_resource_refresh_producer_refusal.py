# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""R2-B1 real service producers refuse unbound refresh before any effect."""

import hashlib
import json

import pytest

from openviking.service.task_tracker import get_task_tracker
from openviking.storage.acl import AclManager, AclMode
from openviking.storage.queuefs.named_queue import NamedQueue
from openviking_cli.exceptions import FailedPreconditionError
from tests.server.conftest import service as service
from tests.server.conftest import temp_dir as temp_dir
from tests.server.test_real_resource_producer_e2e import _records, _root_ctx
from tests.server.test_real_resource_producer_e2e import admitted_service as admitted_service
from tests.storage.test_transfer_merge_binding import _EnabledAclConfig

ROOT = "viking://resources/producer-refusal"
NOTE = ROOT + "/note.md"


async def _seed(service):
    ctx = _root_ctx()
    await service.fs.write(NOTE, "# Original\n\nKeep this body.", ctx=ctx, wait=True)
    fs = service.viking_fs
    store = fs._get_vector_store()
    fs.acl_manager = store.acl_manager = AclManager(store, _EnabledAclConfig())
    await fs.set_acl(
        NOTE,
        [
            {"principal": "user:" + ctx.user.user_id, "level": "manage"},
            {"principal": "user:reader", "level": "read"},
        ],
        ctx=ctx,
        acl_mode=AclMode.RESTRICTED,
    )


async def _snapshot(service):
    fs = service.viking_fs
    ctx = _root_ctx()
    uris = [NOTE, ROOT + "/fresh.md"]
    uris += [
        directory + "/" + name
        for directory in (ROOT, "viking://resources")
        for name in (".abstract.md", ".overview.md")
    ]
    bodies = {}
    for uri in uris:
        bodies[uri] = (
            await fs.read_file_bytes(uri, ctx=ctx) if await fs.exists(uri, ctx=ctx) else None
        )
    records = sorted(
        json.dumps(r, sort_keys=True, default=str) for r in await _records(service, ctx)
    )
    policies = {
        uri: fs.acl_manager.to_report(uri, await fs.acl_manager.resolve(uri, ctx)) for uri in uris
    }
    return bodies, records, policies


def _record_effects(service, monkeypatch):
    """All wrappers call the real native, Task, queue or ACL implementation."""
    fs = service.viking_fs
    calls = []
    targets = [
        (get_task_tracker(), "create"),
        (NamedQueue, "enqueue"),
        (fs.acl_manager, "apply_indexed_update"),
    ]
    targets += [
        (fs._async_agfs, name)
        for name in (
            "write",
            "mkdir",
            "pathlock_acquire_batch",
            "pathlock_acquire_exact_batch",
            "pathlock_acquire_exact",
            "pathlock_acquire_tree",
            "pathlock_to_handoff",
            "pathlock_handoff",
            "pathlock_adopt",
        )
    ]
    for owner, name in targets:
        original = getattr(owner, name)

        async def record(*args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            return await _original(*args, **kwargs)

        monkeypatch.setattr(owner, name, record)
    return calls


async def _leases_available(service, target):
    fs = service.viking_fs
    agfs = fs._async_agfs
    ctx = _root_ctx()
    lease = await agfs.pathlock_acquire_tree(fs._uri_to_path("viking://resources", ctx=ctx))
    await agfs.pathlock_release(lease)
    lease = await agfs.pathlock_acquire_exact(fs._uri_to_path(target, ctx=ctx))
    await agfs.pathlock_release(lease)


@pytest.mark.parametrize("condition", ["create", "replace", "default"])
async def test_resource_batch_refuses_before_effects(admitted_service, monkeypatch, condition):
    service = admitted_service
    await _seed(service)
    before = await _snapshot(service)
    target = ROOT + "/fresh.md" if condition == "create" else NOTE
    operation = {
        "uri": target,
        "content": "must not land",
        "mode": "create" if condition == "create" else "replace",
        "acl": {"mode": "public", "grants": []},
    }
    if condition == "create":
        operation["expected_absent"] = True
    elif condition == "replace":
        operation["expected_sha256"] = hashlib.sha256(before[0][NOTE]).hexdigest()
    calls = _record_effects(service, monkeypatch)
    with pytest.raises(FailedPreconditionError, match="Resource batch refresh is unsupported"):
        await service.fs.batch_write(
            root_uri=ROOT, operations=[operation], ctx=_root_ctx(), wait=True
        )
    assert calls == []
    assert await _snapshot(service) == before
    assert calls == []
    await _leases_available(service, target)


@pytest.mark.parametrize("name", [".abstract.md", ".overview.md"])
async def test_manual_resource_sidecar_refuses_before_effects(admitted_service, monkeypatch, name):
    service = admitted_service
    await _seed(service)
    before = await _snapshot(service)
    calls = _record_effects(service, monkeypatch)
    with pytest.raises(FailedPreconditionError, match="Manual resource semantic sidecar"):
        await service.fs.write(ROOT + "/" + name, "must not land", ctx=_root_ctx(), wait=True)
    assert calls == []
    assert await _snapshot(service) == before
    assert calls == []
    await _leases_available(service, ROOT + "/" + name)


async def test_deferred_conditional_resource_batch_still_writes(admitted_service):
    service = admitted_service
    ctx = _root_ctx()
    await service.fs.write(NOTE, "# Original", ctx=ctx, wait=True)
    before = await _snapshot(service)
    target = ROOT + "/fresh.md"
    result = await service.fs.batch_write(
        root_uri=ROOT,
        operations=[
            {"uri": target, "content": "raw", "mode": "create", "expected_absent": True},
            {
                "uri": NOTE,
                "content": "replaced raw",
                "mode": "replace",
                "expected_sha256": hashlib.sha256(before[0][NOTE]).hexdigest(),
            },
        ],
        defer_processing=True,
        ctx=ctx,
        wait=True,
    )
    assert result["semantic_status"] == result["vector_status"] == "deferred"
    after = await _snapshot(service)
    assert after[0][target] == b"raw"
    assert after[0][NOTE] == b"replaced raw"
    assert {k: v for k, v in before[0].items() if k not in (NOTE, target)} == {
        k: v for k, v in after[0].items() if k not in (NOTE, target)
    }
    assert before[1:] == after[1:]
    await _leases_available(service, target)


async def test_admitted_single_write_keeps_internal_sidecars(admitted_service):
    service = admitted_service
    await service.fs.write(NOTE, "# Admitted\n\nBody.", ctx=_root_ctx(), wait=True)
    bodies, records, _ = await _snapshot(service)
    assert bodies[NOTE] == b"# Admitted\n\nBody."
    assert b"freshness:" in bodies[ROOT + "/.abstract.md"]
    assert bodies[ROOT + "/.overview.md"]
    decoded = [json.loads(record) for record in records]
    assert any(r.get("uri") == NOTE and r.get("level") == 2 for r in decoded)
    assert any(r.get("uri") == ROOT and r.get("level") == 0 for r in decoded)
    await _leases_available(service, NOTE)


async def test_nonresource_batch_and_sidecar_keep_refresh(admitted_service):
    service = admitted_service
    ctx = _root_ctx()
    root = "viking://user/default/memories/producer-control"
    note = root + "/note.md"
    # Lower native helper creates a real existing nonresource sidecar.
    await service.viking_fs.write_file(root + "/.abstract.md", "Original summary", ctx=ctx)
    result = await service.fs.batch_write(
        root_uri=root,
        operations=[{"uri": note, "content": "Control body", "mode": "create"}],
        ctx=ctx,
        wait=True,
    )
    assert result["created"] == [note]
    await service.fs.write(root + "/.abstract.md", "Edited summary", ctx=ctx, wait=True)
    assert "Edited summary" in await service.viking_fs.read_file(root + "/.abstract.md", ctx=ctx)


async def test_file_only_batch_emits_no_refresh(admitted_service):
    service = admitted_service
    ctx = _root_ctx()
    root = "viking://system/memory/memory_templates/custom/producer-control"
    await service.fs.write(root + "/seed.md", "Existing root", ctx=ctx, wait=True)
    note = root + "/note.md"
    result = await service.fs.batch_write(
        root_uri=root,
        operations=[{"uri": note, "content": "Template", "mode": "create"}],
        ctx=ctx,
        wait=True,
    )
    assert result["semantic_status"] == result["vector_status"] == "skipped"
    assert await service.viking_fs.read_file_bytes(note, ctx=ctx) == b"Template"
