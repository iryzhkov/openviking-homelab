"""Original-owner parent refresh after qualified native moves."""

import asyncio
import json

import pytest

from openviking.storage.queuefs.named_queue import NamedQueue
from tests.server.conftest import service as service
from tests.server.conftest import temp_dir as temp_dir
from tests.server.test_real_resource_producer_e2e import (
    _creation_identity,
    _records,
    _root_ctx,
)
from tests.server.test_real_resource_producer_e2e import (
    admitted_service as admitted_service,
)


@pytest.mark.parametrize("different_parent", [False, True])
async def test_no_replace_move_refreshes_under_original_handoffs(
    admitted_service, monkeypatch, different_parent
):
    service = admitted_service
    ctx = _root_ctx()
    source_parent = "viking://resources/move-source"
    target_parent = "viking://resources/move-target" if different_parent else source_parent
    source = source_parent + "/old.md"
    target = target_parent + "/new.md"
    await service.fs.write(source, "# Move this content", ctx=ctx, wait=True, tags=["team=move"])
    if different_parent:
        await service.viking_fs.mkdir(target_parent, ctx=ctx)
    original_identity = await _creation_identity(service)
    raw = await service.viking_fs.read_file_bytes(source, ctx=ctx)
    persisted = []
    original_enqueue = NamedQueue._enqueue

    async def record(self, queue_context):
        persisted.append((self.name, json.loads(json.dumps(queue_context.payload))))
        return await original_enqueue(self, queue_context)

    monkeypatch.setattr(NamedQueue, "_enqueue", record)
    await asyncio.wait_for(service.fs.mv(source, target, ctx=ctx, no_replace=True), timeout=30)
    assert await service.viking_fs.read_file_bytes(target, ctx=ctx) == raw
    assert not await service.viking_fs.exists(source, ctx=ctx)
    semantic = [
        data
        for name, data in persisted
        if "emantic" in name and data.get("generation_trigger") == "content_move"
    ]
    assert {d["uri"] for d in semantic} == {source_parent, target_parent}, persisted
    assert all(d["resource_creation_identity"] == original_identity for d in semantic)
    assert all(d["lock_handoff"] and d["task_id"] for d in semantic)
    records = await _records(service, ctx)
    detail = next(r for r in records if r.get("uri") == target and r.get("level") == 2)
    assert "team=move" in detail["search_tags"], detail
    assert not [r for r in records if r.get("uri") == source]
    for parent in {source_parent, target_parent}:
        assert {(parent, 0), (parent, 1)}.issubset(
            {(r.get("uri"), r.get("level")) for r in records}
        )


async def test_move_account_change_refuses_before_parent_metadata(admitted_service, monkeypatch):
    from openviking.service.fs_service import FSService
    from tests.server.test_real_resource_producer_e2e import _recreate_account

    service = admitted_service
    source = "viking://resources/move-refuse/old.md"
    target = "viking://resources/move-refuse/new.md"
    await service.fs.write(source, "# Original", ctx=_root_ctx(), wait=True)
    original = FSService._enqueue_owned_move_refresh
    pending = []
    changed = []

    async def record_pending(**kwargs):
        pending.append(True)
        raise AssertionError("Changed account must refuse before pending writes")

    async def recreate_then_refresh(self, **kwargs):
        if not changed:
            await _recreate_account(service)
            changed.append(True)
        return await original(self, **kwargs)

    monkeypatch.setattr(
        "openviking.service.fs_service.mark_abstract_overview_pending", record_pending
    )
    monkeypatch.setattr(FSService, "_enqueue_owned_move_refresh", recreate_then_refresh)
    with pytest.raises(Exception, match="lifetime|identity|changed"):
        await asyncio.wait_for(service.fs.mv(source, target, ctx=_root_ctx(), no_replace=True), 30)
    assert changed and not pending
    # The caller received refusal after the already committed move.
    assert await service.viking_fs.read_file_bytes(target, ctx=_root_ctx())


async def test_move_caller_cancellation_finishes_parent_work(admitted_service, monkeypatch):
    from openviking.service.fs_service import FSService
    from openviking.service.task_tracker import TaskStatus, get_task_tracker

    service = admitted_service
    ctx = _root_ctx()
    source = "viking://resources/move-cancel/old.md"
    target = "viking://resources/move-cancel/new.md"
    await service.fs.write(source, "# Cancelled caller", ctx=ctx, wait=True)
    reached = asyncio.Event()
    proceed = asyncio.Event()
    original = FSService._enqueue_owned_move_refresh
    task_ids = []
    enqueue = NamedQueue._enqueue

    async def record(self, queue_context):
        if queue_context.payload.get("generation_trigger") == "content_move":
            task_ids.append(queue_context.payload["task_id"])
        return await enqueue(self, queue_context)

    async def pause(self, **kwargs):
        reached.set()
        await proceed.wait()
        return await original(self, **kwargs)

    monkeypatch.setattr(NamedQueue, "_enqueue", record)
    monkeypatch.setattr(FSService, "_enqueue_owned_move_refresh", pause)
    caller = asyncio.create_task(service.fs.mv(source, target, ctx=ctx, no_replace=True))
    await asyncio.wait_for(reached.wait(), 10)
    caller.cancel()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, 30)
    assert task_ids
    task = await get_task_tracker().get(
        task_ids[0], account_id=ctx.account_id, user_id=ctx.user.user_id
    )
    assert task.status is TaskStatus.COMPLETED
    records = await _records(service, ctx)
    assert (target, 2) in {(r.get("uri"), r.get("level")) for r in records}
