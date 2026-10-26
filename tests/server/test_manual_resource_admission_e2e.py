"""Manual write admission through real persisted queues and handlers."""

import asyncio
import json

import pytest

from openviking.storage.queuefs.named_queue import NamedQueue
from tests.server.conftest import service as service
from tests.server.conftest import temp_dir as temp_dir
from tests.server.test_real_resource_producer_e2e import (
    _creation_identity,
    _records,
    _recreate_account,
    _root_ctx,
)
from tests.server.test_real_resource_producer_e2e import (
    admitted_service as admitted_service,
)


async def test_manual_resource_write_persisted_semantic_vectors(admitted_service, monkeypatch):
    service = admitted_service
    ctx = _root_ctx()
    original = await _creation_identity(service)
    persisted = []
    enqueue = NamedQueue.enqueue

    async def record(self, data):
        persisted.append((self.name, json.loads(json.dumps(data))))
        return await enqueue(self, data)

    monkeypatch.setattr(NamedQueue, "enqueue", record)
    uri = "viking://resources/manual/note.md"
    result = await asyncio.wait_for(
        service.fs.write(uri, "# Fresh manual note\n\nAdmission test content.", ctx=ctx, wait=True),
        timeout=25,
    )
    semantic = [
        d for n, d in persisted if "emantic" in n and d.get("generation_trigger") == "content_write"
    ]
    assert semantic, persisted
    assert all(d["resource_creation_identity"] == original for d in semantic)
    assert all(d["lock_handoff"] for d in semantic)
    assert result["semantic_status"] == "complete", result
    records = await _records(service, ctx)
    assert {
        (uri, 2),
        ("viking://resources/manual", 0),
        ("viking://resources/manual", 1),
    }.issubset({(r.get("uri"), r.get("level")) for r in records}), records


async def test_manual_write_recreated_before_commit_leaves_no_file(admitted_service, monkeypatch):
    service = admitted_service
    fs = service.viking_fs
    acquire = fs._acquire_resource_coverage
    changed = False

    async def recreate(ctx, admitted, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            await _recreate_account(service)
        return await acquire(ctx, admitted, **kwargs)

    monkeypatch.setattr(fs, "_acquire_resource_coverage", recreate)
    uri = "viking://resources/manual-recreated.md"
    with pytest.raises(Exception, match="identity|recreated|changed"):
        await service.fs.write(uri, "Must not commit", ctx=_root_ctx(), wait=False)
    assert changed
    from openviking_cli.exceptions import NotFoundError

    with pytest.raises(NotFoundError):
        await fs.stat(uri, ctx=_root_ctx())


async def test_manual_write_recreated_before_job_refuses_semantic(admitted_service, monkeypatch):
    service = admitted_service
    from openviking.storage.queuefs.content_write_processor import ContentWriteProcessor

    original = ContentWriteProcessor._on_dequeue_bound
    changed = []

    async def recreate(self, data):
        if not changed:
            await _recreate_account(service)
            changed.append(True)
        return await original(self, data)

    monkeypatch.setattr(ContentWriteProcessor, "_on_dequeue_bound", recreate)
    with pytest.raises(Exception, match="identity|recreated|changed"):
        await asyncio.wait_for(
            service.fs.write("viking://resources/stale.md", "# Stale", ctx=_root_ctx(), wait=True),
            timeout=25,
        )
    assert changed
    records = await _records(service, _root_ctx())
    assert not [r for r in records if str(r.get("uri", "")).startswith("viking://resources/stale")]


async def test_reindex_entry_preserves_original_admission(admitted_service, monkeypatch):
    service = admitted_service
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    ctx = _root_ctx()
    uri = "viking://resources/reindex-fresh"
    await service.viking_fs.mkdir(uri, ctx=ctx)
    await service.viking_fs.write_file(uri + "/note.md", "# Reindex fresh body", ctx=ctx)
    original = await _creation_identity(service)
    persisted = []
    enqueue = NamedQueue.enqueue

    async def record(self, data):
        persisted.append((self.name, json.loads(json.dumps(data))))
        return await enqueue(self, data)

    monkeypatch.setattr(NamedQueue, "enqueue", record)
    result = await asyncio.wait_for(
        service.reindex(uri=uri, mode="semantic_and_vectors", wait=True, ctx=ctx),
        timeout=25,
    )
    jobs = [d for n, d in persisted if "eindex" in n]
    semantic = [d for n, d in persisted if "emantic" in n]
    assert jobs and semantic, persisted
    assert all(d["resource_creation_identity"] == original for d in jobs + semantic)
    assert result["failed_records"] == 0, result
    assert (uri + "/note.md", 2) in {
        (r.get("uri"), r.get("level")) for r in await _records(service, ctx)
    }


async def test_legacy_content_job_refuses_without_identity_recapture(admitted_service, monkeypatch):
    from openviking.service.task_tracker import get_task_tracker
    from openviking.storage.queuefs.content_write_msg import ContentWriteMsg
    from openviking.storage.queuefs.content_write_processor import ContentWriteProcessor

    ctx = _root_ctx()
    service = admitted_service

    async def forbidden(*args, **kwargs):
        raise AssertionError("Legacy job tried to mint admission or adopt a lock")

    task = await get_task_tracker().create(
        "content_write",
        account_id=ctx.account_id,
        user_id=ctx.user.user_id,
        resource_id="viking://resources/legacy.md",
    )
    msg = ContentWriteMsg(
        task_id=task.task_id,
        operation="content_write",
        resource_id="viking://resources/legacy.md",
        result={},
        refresh={"kind": "plan"},
        account_id=ctx.account_id,
        user_id=ctx.user.user_id,
        role=str(ctx.role),
    )
    monkeypatch.setattr(service.viking_fs, "_capture_resource_identity", forbidden)
    monkeypatch.setattr(service.viking_fs._async_agfs, "pathlock_adopt", forbidden)
    result = await ContentWriteProcessor(service.viking_fs).on_dequeue(msg.to_dict())
    assert result.outcome.value == "failed"
    assert "original admission" in result.error


async def test_lost_content_owner_never_reacquires_or_cleans_foreign(admitted_service):
    from openviking.storage.queuefs.content_write_msg import ContentWriteMsg
    from openviking.storage.queuefs.content_write_processor import ContentWriteProcessor

    fs = admitted_service.viking_fs
    ctx = _root_ctx()
    identity = await fs._capture_resource_identity(ctx)
    path = fs._uri_to_path("viking://resources/lost-owner", ctx=ctx)
    lease = await fs._async_agfs.pathlock_acquire_tree(path)
    handoff = await fs._async_agfs.pathlock_to_handoff(lease)
    await fs._async_agfs.pathlock_release(lease)
    foreign = await fs._async_agfs.pathlock_acquire_tree(path)
    msg = ContentWriteMsg(
        task_id="lost-owner",
        operation="content_write",
        resource_id="viking://resources/lost-owner/note.md",
        result={},
        refresh={
            "kind": "plan",
            "resource_admission": {
                "creation_identity": identity.creation_identity,
                "owner": lease,
                "index_conditions": {},
            },
        },
        account_id=ctx.account_id,
        user_id=ctx.user.user_id,
        role=str(ctx.role),
        lock_handoff=handoff,
    )
    try:
        from openviking.storage.errors import LockAcquisitionError

        with pytest.raises((RuntimeError, ValueError, LockAcquisitionError)):
            await ContentWriteProcessor(fs)._load_lock(msg, ctx)
        assert await fs._async_agfs.pathlock_refresh(foreign) == "refreshed"
    finally:
        await fs._async_agfs.pathlock_release(foreign)
