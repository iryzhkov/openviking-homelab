# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Fresh ResourceService admission through persisted native AddResource delivery."""

import asyncio
import json
from uuid import uuid4

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.queuefs import QueueManager, get_queue_manager
from openviking.storage.queuefs.add_resource_msg import AddResourceMsg
from openviking.storage.queuefs.process_result import ProcessOutcome
from openviking_cli.exceptions import ConflictError, InvalidArgumentError
from tests.server import test_real_resource_producer_e2e as producer
from tests.server.test_real_resource_producer_e2e import _creation_identity, _records, _root_ctx

admitted_service = producer.admitted_service
trace = producer.trace


async def _change_lifetime(svc, change):
    from openviking.server.api_keys.legacy import ACCOUNTS_PATH

    agfs = svc.viking_fs._async_agfs
    data = json.loads(await agfs.cat(ACCOUNTS_PATH, fs_ctx={"bypass_cache": "true"}))
    record = data["accounts"][_root_ctx().account_id]
    if change == "recreate":
        record["creation_id"] = str(uuid4())
    elif change == "delete":
        del data["accounts"][_root_ctx().account_id]
    elif change == "fence":
        record["deletion"] = {
            "task_id": "test-fence",
            "owner_account_id": "default",
            "owner_user_id": "default",
        }
    await agfs.write(ACCOUNTS_PATH, json.dumps(data).encode())


def _payload(raw):
    data = raw.get("data", raw)
    return json.loads(data) if isinstance(data, str) else data


async def _park(svc, source, ctx, *, monkeypatch=None, corrupt=None):
    await asyncio.wait_for(get_queue_manager().wait_complete(), timeout=15)
    queue = get_queue_manager().get_queue(QueueManager.ADD_RESOURCE)
    handler = queue._dequeue_handler
    queue.set_dequeue_handler(None)
    if corrupt is not None:
        original_enqueue = queue._enqueue

        async def corrupt_before_persistence(enqueue_ctx):
            corrupt(enqueue_ctx.payload)
            return await original_enqueue(enqueue_ctx)

        monkeypatch.setattr(queue, "_enqueue", corrupt_before_persistence)
    result = await svc.resources.add_resource(
        path=str(source), to="viking://resources/admission-case", ctx=ctx, wait=False
    )
    raw = await queue.dequeue_raw()
    assert raw and _payload(raw)["job_phase"] == "source", result
    if corrupt is None:
        assert _payload(raw)["resource_creation_identity"] == await _creation_identity(svc)
    return queue, handler, raw


@pytest.mark.parametrize("role", [Role.ROOT, Role.ADMIN])
@pytest.mark.parametrize("change", ["recreate", "delete", "fence"])
async def test_old_persisted_request_refuses_before_resource_mutation(
    admitted_service, sample_markdown_file, monkeypatch, role, change
):
    svc = admitted_service
    ctx = RequestContext(user=_root_ctx().user, role=role)
    queue, handler, raw = await _park(svc, sample_markdown_file, ctx)
    await _change_lifetime(svc, change)
    # Model the recreated lifetime already owning bytes in the same namespace.
    fs = svc.viking_fs
    root = fs._uri_to_path("viking://resources", ctx=ctx)
    sentinel = root + "/new-lifetime.txt"
    await fs._async_agfs.write(sentinel, b"new account owns these bytes")
    before = await _records(svc, ctx)
    effects = []
    for name in ("write", "mkdir", "rm", "mv"):
        original = getattr(fs._async_agfs, name)

        async def observe(*args, _name=name, _original=original, **kwargs):
            if any(isinstance(arg, str) and arg.startswith(root) for arg in args):
                effects.append((_name, args))
            return await _original(*args, **kwargs)

        monkeypatch.setattr(fs._async_agfs, name, observe)

    async def no_fresh_capture(*args, **kwargs):
        pytest.fail("old queue work called fresh admission fallback")

    monkeypatch.setattr(fs, "_capture_resource_identity", no_fresh_capture)
    queue.set_dequeue_handler(handler)
    result = await queue.process_dequeued(raw)
    assert result.outcome is ProcessOutcome.FAILED, result
    assert not effects, effects
    assert await fs._async_agfs.cat(sentinel) == b"new account owns these bytes"
    assert await _records(svc, ctx) == before
    await queue.ack(raw["id"], raw)


@pytest.mark.parametrize("change", ["recreate", "delete", "fence"])
async def test_account_changes_after_real_owner_adoption(
    admitted_service, sample_markdown_file, monkeypatch, trace, change
):
    svc = admitted_service
    ctx = _root_ctx()
    queue, handler, raw = await _park(svc, sample_markdown_file, ctx)
    original_load = handler._load_lock
    adopted = []

    async def change_after_adoption(msg, actual_ctx):
        lease = await original_load(msg, actual_ctx)
        assert lease
        adopted.append(lease)
        await _change_lifetime(svc, change)
        return lease

    monkeypatch.setattr(handler, "_load_lock", change_after_adoption)
    before = await _records(svc, ctx)
    queue.set_dequeue_handler(handler)
    result = await queue.process_dequeued(raw)
    assert adopted
    assert result.outcome is ProcessOutcome.FAILED, result
    assert not trace[0], trace[0]
    assert await _records(svc, ctx) == before
    await queue.ack(raw["id"], raw)


async def test_cancellation_rechecks_original_after_real_owner_adoption(
    admitted_service, sample_markdown_file, monkeypatch
):
    svc = admitted_service
    ctx = _root_ctx()
    queue, handler, raw = await _park(svc, sample_markdown_file, ctx)
    agfs = svc.viking_fs._async_agfs
    original_adopt = agfs.pathlock_adopt
    original_cleanup = svc.resources._cleanup_reserved_target_if_empty
    adopted = []
    cleanup_calls = []

    async def change_after_adoption(*args, **kwargs):
        lease = await original_adopt(*args, **kwargs)
        adopted.append(lease)
        await _change_lifetime(svc, "recreate")
        return lease

    async def observe_cleanup(*args, **kwargs):
        cleanup_calls.append(kwargs)
        return await original_cleanup(*args, **kwargs)

    monkeypatch.setattr(agfs, "pathlock_adopt", change_after_adoption)
    monkeypatch.setattr(svc.resources, "_cleanup_reserved_target_if_empty", observe_cleanup)
    before = await _records(svc, ctx)
    result = await handler.on_cancelled(raw)
    assert result.outcome is ProcessOutcome.CANCELLED, result
    assert adopted and not cleanup_calls
    assert await _records(svc, ctx) == before
    await queue.ack(raw["id"], raw)


@pytest.mark.parametrize("bad", [None, "", " ", 7, True, {}, []])
async def test_legacy_or_malformed_job_cannot_recapture(
    admitted_service, sample_markdown_file, monkeypatch, bad
):
    svc = admitted_service

    def corrupt(data):
        if bad is None:
            data.pop("resource_creation_identity")
        else:
            data["resource_creation_identity"] = bad

    queue, handler, raw = await _park(
        svc, sample_markdown_file, _root_ctx(), monkeypatch=monkeypatch, corrupt=corrupt
    )
    # Fault injection precedes NamedQueue's real native persistence.
    before = await _records(svc, _root_ctx())

    async def no_fresh_capture(*args, **kwargs):
        pytest.fail("legacy queued intent called fresh admission fallback")

    monkeypatch.setattr(svc.viking_fs, "_capture_resource_identity", no_fresh_capture)
    queue.set_dequeue_handler(handler)
    result = await queue.process_dequeued(raw)
    assert result.outcome is ProcessOutcome.FAILED, result
    assert await _records(svc, _root_ctx()) == before
    await queue.ack(raw["id"], raw)


async def test_fresh_service_request_preserves_original_through_semantic(
    admitted_service, sample_markdown_file, trace, monkeypatch
):
    svc = admitted_service
    original = await _creation_identity(svc)
    capture = svc.viking_fs._capture_resource_identity
    captured = []

    async def record_capture(ctx):
        identity = await capture(ctx)
        captured.append(identity.creation_identity)
        return identity

    monkeypatch.setattr(svc.viking_fs, "_capture_resource_identity", record_capture)
    result = await svc.resources.add_resource(
        path=str(sample_markdown_file), ctx=_root_ctx(), wait=True
    )
    assert result.get("status") == "success", result
    calls, persisted = trace
    assert "process_resource" in calls
    source = [d for name, d in persisted if name == "AddResource"]
    semantic = [d for name, d in persisted if name == "Semantic"]
    assert source and semantic
    assert captured == [original], "queued execution must not use fresh capture fallback"
    assert all(d["resource_creation_identity"] == original for d in source + semantic)
    root = result["root_uri"]
    records = {(r["uri"], r["level"]) for r in await _records(svc, _root_ctx())}
    assert {(root, 0), (root, 1), (root + "/sample.md", 2)} <= records


async def test_scheduled_watch_intent_cannot_receive_fresh_admission(
    admitted_service, sample_markdown_file, monkeypatch
):
    async def no_fresh_capture(*args, **kwargs):
        pytest.fail("cached WatchTask intent received fresh account authority")

    monkeypatch.setattr(admitted_service.viking_fs, "_capture_resource_identity", no_fresh_capture)
    with pytest.raises(ConflictError, match="WatchTask"):
        await admitted_service.resources.refresh_resource(
            path=str(sample_markdown_file), ctx=_root_ctx(), to="viking://resources/watched"
        )


@pytest.mark.parametrize("where", ["top", "args"])
async def test_public_caller_cannot_supply_internal_admission(
    admitted_service, sample_markdown_file, where
):
    kwargs = {"resource_creation_identity": "forged"}
    if where == "args":
        kwargs = {"args": kwargs}
    with pytest.raises(InvalidArgumentError, match="resource_creation_identity"):
        await admitted_service.resources.add_resource(
            path=str(sample_markdown_file), ctx=_root_ctx(), **kwargs
        )


async def test_explicit_nonresource_request_keeps_existing_pipeline(
    admitted_service, sample_markdown_file, monkeypatch
):
    svc = admitted_service

    async def no_resource_capture(*args, **kwargs):
        pytest.fail("explicit nonresource request acquired resource admission")

    monkeypatch.setattr(svc.viking_fs, "_capture_resource_identity", no_resource_capture)
    result = await svc.resources.add_resource(
        path=str(sample_markdown_file),
        ctx=_root_ctx(),
        to="viking://user/default/memories/admission-control",
        wait=True,
        build_index=False,
    )
    assert result.get("status") == "success", result
    assert result["root_uri"] == "viking://user/default/memories/admission-control"


def test_nonresource_job_keeps_optional_admission_schema():
    msg = AddResourceMsg(
        task_id="non-resource",
        root_uri="viking://user/u/memories/x",
        account_id="a",
        user_id="u",
        role="user",
        path="text",
    )
    assert AddResourceMsg.from_dict(msg.to_dict()).resource_creation_identity is None
