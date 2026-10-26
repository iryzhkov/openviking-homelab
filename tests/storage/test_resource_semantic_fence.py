# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Resource queue fences exercised with the native lease manager."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from openviking.pyagfs import AsyncAGFSClient
from openviking.storage.context_update_plan import (
    ContentState,
    SemanticAction,
    SemanticPlan,
    SemanticTreeEntry,
    SemanticTreeSnapshot,
)
from openviking.storage.errors import LockAcquisitionError
from openviking.storage.queuefs.process_result import ProcessOutcome
from openviking.storage.queuefs.semantic_executor import SemanticTreeExecutor, SemanticTreeStats
from openviking.storage.queuefs.semantic_msg import SemanticMsg
from openviking.storage.queuefs.semantic_processor import SemanticProcessor
from openviking.storage.viking_fs import SyncDiff
from openviking.utils.content_hash import content_md5


@pytest.fixture
def env(tmp_path, monkeypatch):
    import ragfs_python

    from openviking.service.task_store import PersistentTaskStore
    from openviking.service.task_tracker import TaskTracker
    from tests.test_task_tracker import _FakeAgfs

    monkeypatch.setattr(
        "openviking.service.task_tracker._instance", TaskTracker(PersistentTaskStore(_FakeAgfs()))
    )
    client = ragfs_python.RAGFSBindingClient()
    client.mount("localfs", "/local", {"local_dir": str(tmp_path)})
    agfs = AsyncAGFSClient(client)
    resources = tmp_path / "account/resources"
    resources.mkdir(parents=True)
    target = resources / "document"
    target.mkdir()
    content = target / "content.md"
    content.write_bytes(b"original")
    foreign = target / "foreign-empty"
    foreign.mkdir()
    root_uri = "viking://resources/document"

    def uri_path(uri, ctx=None):
        del ctx
        assert uri.startswith("viking://resources")
        return "/local/account/resources" + uri[len("viking://resources") :]

    def disk_path(uri):
        return tmp_path / uri_path(uri).removeprefix("/local/")

    from openviking.server.account_stores.models import AccountIdentity
    from openviking.service.account_identity import capture_account_identity

    manager = SimpleNamespace(
        get_account_identity=AsyncMock(return_value=AccountIdentity("test-creation", False))
    )
    fs = SimpleNamespace(
        _account_identity_manager=manager,
        _capture_resource_identity=lambda ctx: capture_account_identity(manager, ctx.account_id),
        _async_agfs=agfs,
        _uri_to_path=uri_path,
        exists=AsyncMock(side_effect=lambda uri, **kwargs: disk_path(uri).exists()),
        read_file_bytes=AsyncMock(side_effect=lambda uri, **kwargs: disk_path(uri).read_bytes()),
    )
    for module in ("semantic_processor", "semantic_work", "semantic_lock", "semantic_executor"):
        monkeypatch.setattr(f"openviking.storage.queuefs.{module}.get_viking_fs", lambda: fs)
    processor = SemanticProcessor(vlm_resolver=SimpleNamespace(get_vlm=AsyncMock()))
    processor._enqueue_parent_refresh = AsyncMock()
    writes = []

    async def sync(*args, **kwargs):
        content.write_bytes(b"queued old material")
        writes.append("sync")
        return SyncDiff()

    processor._sync_topdown_recursive = AsyncMock(side_effect=sync)

    async def run(uri):
        validator = constructor.call_args.kwargs.get("before_resource_write")
        if validator is not None:
            await validator()
        writes.append(uri)
        content.write_bytes(b"queued old material")

    executor = SimpleNamespace(
        run=AsyncMock(side_effect=run),
        get_stats=Mock(return_value=SemanticTreeStats()),
        stale=False,
    )
    constructor = Mock(return_value=executor)
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.SemanticTreeExecutor", constructor
    )

    def message(**kwargs):
        # Resource messages carry the original request identity captured by
        # their producer; processing never captures a fresh one.
        kwargs.setdefault("resource_creation_identity", "test-creation")
        return SemanticMsg(
            uri=root_uri,
            context_type="resource",
            account_id="account",
            skip_vectorization=True,
            propagate_to_parent=False,
            **kwargs,
        )

    return SimpleNamespace(
        agfs=agfs,
        fs=fs,
        processor=processor,
        executor=executor,
        constructor=constructor,
        resources=resources,
        target=target,
        content=content,
        foreign=foreign,
        root_uri=root_uri,
        message=message,
        writes=writes,
    )


async def handoff(env):
    lease = await env.agfs.pathlock_acquire_tree("/local/account/resources/document")
    token = await env.agfs.pathlock_to_handoff(lease)
    await env.agfs.pathlock_handoff(lease)
    return token


@pytest.mark.parametrize("payload", ["source", "plan", "artifact", "content"])
async def test_stale_handoff_preserves_intervening_replacement(env, payload):
    token = await handoff(env)
    # Another consumer settled the original ownership. The old durable message
    # still exists, but its handoff is now stale.
    adopted = await env.agfs.pathlock_adopt(token)
    await env.agfs.pathlock_release(adopted)
    replacement = await env.agfs.pathlock_acquire_tree("/local/account/resources")
    try:
        env.content.write_bytes(b"replacement must survive")
        (env.target / ".abstract.md").write_text("new sidecar")
    finally:
        await env.agfs.pathlock_release(replacement)
    kwargs = {"lock_handoff": token}
    if payload == "source":
        kwargs["target_uri"] = env.root_uri + "/other"
    elif payload == "artifact":
        kwargs["artifact_files"] = ["content.md"]
    elif payload == "content":
        kwargs.update(
            generation_trigger="content_write",
            file_md5s={env.root_uri + "/content.md": content_md5(b"original")},
        )
    else:
        kwargs["plan"] = SemanticPlan(
            env.root_uri,
            "resource",
            SemanticTreeSnapshot(
                (
                    SemanticTreeEntry(
                        "", "directory", ContentState.UNCHANGED, SemanticAction.AGGREGATE
                    ),
                )
            ),
        )
    result = await env.processor.on_dequeue(env.message(**kwargs).to_dict())
    assert env.content.read_bytes() == b"replacement must survive"
    assert result.outcome is ProcessOutcome.FAILED
    assert "ownership was lost" in result.error
    env.constructor.assert_not_called()
    assert env.content.read_bytes() == b"replacement must survive"
    assert (env.target / ".abstract.md").read_text() == "new sidecar"
    assert env.foreign.is_dir()
    lease = await env.agfs.pathlock_acquire_tree("/local/account/resources")
    await env.agfs.pathlock_release(lease)


async def test_nonresource_scope_keeps_existing_stale_recovery(env):
    from openviking.storage.queuefs.semantic_lock import SemanticLockScope

    token = await handoff(env)
    adopted = await env.agfs.pathlock_adopt(token)
    await env.agfs.pathlock_release(adopted)
    scope = await SemanticLockScope.resolve(token)
    assert scope.lock is not None
    await scope.close()


@pytest.mark.parametrize("planned", [False, True])
async def test_live_handoff_still_runs_and_releases(env, planned):
    token = await handoff(env)
    plan = (
        SemanticPlan(
            env.root_uri,
            "resource",
            SemanticTreeSnapshot(
                (
                    SemanticTreeEntry(
                        "", "directory", ContentState.UNCHANGED, SemanticAction.AGGREGATE
                    ),
                )
            ),
        )
        if planned
        else None
    )
    result = await env.processor.on_dequeue(env.message(lock_handoff=token, plan=plan).to_dict())
    assert result.outcome is ProcessOutcome.SUCCESS
    assert env.writes == [env.root_uri]
    assert env.content.read_bytes() == b"queued old material"
    lease = await env.agfs.pathlock_acquire_tree("/local/account/resources")
    await env.agfs.pathlock_release(lease)


@pytest.mark.parametrize("qualified", [False, True])
async def test_no_handoff_cached_file_compares_current_bytes_under_tree(env, qualified):
    uri = env.root_uri + "/content.md"
    msg = env.message(
        generation_trigger="content_write",
        changes={"modified": [uri]},
        file_abstracts={uri: "cached original summary"},
        file_md5s={uri: content_md5(b"original")} if qualified else {},
    )
    # This is the intervening content change, after producer enqueue.
    env.content.write_bytes(b"new bytes")
    original_read = env.fs.read_file_bytes

    async def read_under_coverage(uri, **kwargs):
        with pytest.raises(LockAcquisitionError):
            await env.agfs.pathlock_acquire_exact("/local/account/resources/contender")
        return await original_read(uri, **kwargs)

    env.fs.read_file_bytes = AsyncMock(side_effect=read_under_coverage)
    result = await env.processor.on_dequeue(msg.to_dict())
    assert result.outcome is ProcessOutcome.FAILED
    assert ("bytes changed" if qualified else "lack byte compare") in result.error
    assert env.content.read_bytes() == b"new bytes"
    assert env.foreign.is_dir()
    env.constructor.assert_not_called()
    lease = await env.agfs.pathlock_acquire_tree("/local/account/resources")
    await env.agfs.pathlock_release(lease)


async def test_matching_md5_is_a_real_compare_guard(env):
    uri = env.root_uri + "/content.md"
    msg = env.message(
        generation_trigger="content_write",
        changes={"modified": [uri]},
        file_md5s={uri: content_md5(b"original")},
    )
    result = await env.processor.on_dequeue(msg.to_dict())
    assert result.outcome is ProcessOutcome.SUCCESS
    env.fs.read_file_bytes.assert_awaited_once()
    assert env.writes == [env.root_uri]


@pytest.mark.parametrize(
    "cached",
    [
        {"target_uri": "viking://resources/other"},
        {"artifact_ref": {"backend": "local"}},
        {"artifact_files": ["content.md"]},
        {"copy_source_uri": "viking://resources/old"},
        {"source": {"path": "historical"}},
    ],
)
async def test_unsupported_legacy_cached_work_fails_visibly_without_writes(env, cached):
    result = await env.processor.on_dequeue(env.message(**cached).to_dict())
    assert result.outcome is ProcessOutcome.FAILED
    assert "require a live handoff" in result.error
    env.constructor.assert_not_called()
    assert env.content.read_bytes() == b"original"
    assert env.foreign.is_dir()


async def test_no_handoff_refresh_owns_coverage_across_execution(env):
    async def run(uri):
        with pytest.raises(LockAcquisitionError):
            await env.agfs.pathlock_acquire_exact("/local/account/resources/sibling")
        assert uri == env.root_uri
        lock = env.constructor.call_args.kwargs["lock"]
        coverage = await env.agfs.pathlock_to_handoff(lock)
        assert coverage["covered_paths"]

    env.executor.run.side_effect = run
    result = await env.processor.on_dequeue(env.message().to_dict())
    assert result.outcome is ProcessOutcome.SUCCESS
    lease = await env.agfs.pathlock_acquire_tree("/local/account/resources")
    await env.agfs.pathlock_release(lease)


async def test_lost_lease_blocks_cached_sidecar_and_file_vector(env, monkeypatch):
    from openviking.storage.queuefs.semantic_lock import ResourceInputConflict
    from openviking.storage.queuefs.semantic_work import ResourceSemanticMessageWork

    msg = env.message()
    work = ResourceSemanticMessageWork(env.processor, msg, None)
    ctx = env.processor._ctx_from_semantic_msg(msg)
    await work.acquire_lock(ctx)
    await env.agfs.pathlock_release(work.scope.lock)
    sidecars = AsyncMock()
    vectors = AsyncMock()
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_executor.write_abstract_overview", sidecars
    )
    processor = SimpleNamespace(
        _generate_single_file_summary=AsyncMock(return_value={"summary": "old"}),
        _vectorize_single_file=vectors,
    )
    executor = SemanticTreeExecutor(
        processor=processor,
        context_type="resource",
        max_concurrent_llm=1,
        ctx=ctx,
        lock=work.scope.lock,
        before_resource_write=work.validate_ownership,
    )
    with pytest.raises(ResourceInputConflict, match="ownership was lost"):
        await executor._write_directory_semantics(
            env.root_uri,
            "old overview",
            "old abstract",
            total_entries=1,
            sampled_entries=1,
            consume_pending=0,
        )
    with pytest.raises(ResourceInputConflict, match="ownership was lost"):
        await executor._file_summary_task(env.root_uri, env.root_uri + "/content.md")
    sidecars.assert_not_awaited()
    vectors.assert_not_awaited()
    assert env.content.read_bytes() == b"original"
    await work.close()


async def test_native_ownership_loss_preserves_foreign_token_and_directory(env):
    from openviking.storage.queuefs.semantic_lock import ResourceInputConflict
    from openviking.storage.queuefs.semantic_work import ResourceSemanticMessageWork

    msg = env.message()
    work = ResourceSemanticMessageWork(env.processor, msg, None)
    await work.acquire_lock(env.processor._ctx_from_semantic_msg(msg))
    lock_file = env.resources / ".path.ovlock"
    _, timestamp, kind = lock_file.read_text().split(":")
    foreign_token = f"foreign-owner:{timestamp}:{kind}"
    lock_file.write_text(foreign_token)
    with pytest.raises(ResourceInputConflict, match="ownership was lost"):
        await work.validate_ownership()
    from openviking.pyagfs.exceptions import AGFSInternalError

    # A foreign token is a visible cleanup failure, never permission to remove it.
    with pytest.raises(AGFSInternalError, match="changed while releasing"):
        await work.close()
    assert lock_file.read_text() == foreign_token
    assert env.foreign.is_dir()
    assert env.content.read_bytes() == b"original"


@pytest.mark.parametrize("cancel", [False, True])
async def test_taskless_refresh_collects_and_drains_vector_work_before_unlock(
    env, monkeypatch, cancel
):
    from openviking.service.task_queue_middleware import TaskWorkQueueMiddleware
    from openviking.service.task_work_index import get_task_context
    from openviking.storage.queuefs.named_queue import NamedQueue
    from openviking.storage.queuefs.process_result import ProcessResult

    backend = SimpleNamespace(mkdir=AsyncMock(), write=AsyncMock(return_value="vector-message"))
    monkeypatch.setattr("openviking.storage.queuefs.named_queue.AsyncAGFSClient", lambda _: backend)
    queue = NamedQueue(
        object(),
        "/queue",
        "Embedding",
        dequeue_handler=SimpleNamespace(on_dequeue=AsyncMock(return_value=ProcessResult.success())),
        middlewares=[TaskWorkQueueMiddleware()],
    )
    enqueued = asyncio.Event()

    async def run(uri):
        assert get_task_context() is not None
        await queue.enqueue(
            {"context_data": {"uri": uri, "account_id": "account", "user_id": "default"}}
        )
        enqueued.set()

    env.executor.run.side_effect = run
    msg = env.message()
    msg.skip_vectorization = False
    task = asyncio.create_task(env.processor.on_dequeue(msg.to_dict()))
    await asyncio.wait_for(enqueued.wait(), timeout=2)
    if cancel:
        task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    with pytest.raises(LockAcquisitionError):
        await env.agfs.pathlock_acquire_exact("/local/account/resources/contender")
    payload = backend.write.call_args.args[1].decode()
    backend.read = AsyncMock(
        side_effect=[
            json.dumps({"id": "vector-message", "data": payload}).encode(),
            json.dumps({"pending": 0, "processing": 0}).encode(),
        ]
    )
    await queue.dequeue()
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert (await task).outcome is ProcessOutcome.SUCCESS
    lease = await env.agfs.pathlock_acquire_tree("/local/account/resources")
    await env.agfs.pathlock_release(lease)


async def test_resource_generation_failures_are_not_successful_indexing(env):
    env.executor.get_stats.return_value = SemanticTreeStats(failures=["sidecar rejected"])
    result = await env.processor.on_dequeue(env.message().to_dict())
    assert result.outcome is ProcessOutcome.FAILED
    assert "sidecar rejected" in result.error


async def test_cancellation_drains_execution_before_releasing_ownership(env):
    started = asyncio.Event()
    finished = asyncio.Event()

    async def run(uri):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    env.executor.run.side_effect = run
    task = asyncio.create_task(env.processor.on_dequeue(env.message().to_dict()))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()
    lease = await env.agfs.pathlock_acquire_tree("/local/account/resources")
    await env.agfs.pathlock_release(lease)


async def test_refusal_marks_actual_task_and_queue_failed(env, monkeypatch):
    from openviking.service.task_queue_middleware import TaskWorkQueueMiddleware
    from openviking.service.task_store import PersistentTaskStore
    from openviking.service.task_tracker import TaskStatus, TaskTracker
    from openviking.service.task_work_index import bind_task_context
    from openviking.storage.queuefs.named_queue import NamedQueue
    from tests.test_task_tracker import _FakeAgfs

    tracker = TaskTracker(PersistentTaskStore(_FakeAgfs()))
    monkeypatch.setattr("openviking.service.task_tracker._instance", tracker)
    backend = SimpleNamespace(mkdir=AsyncMock(), write=AsyncMock(return_value="queue-message"))
    monkeypatch.setattr("openviking.storage.queuefs.named_queue.AsyncAGFSClient", lambda _: backend)
    queue = NamedQueue(
        object(),
        "/queue",
        "Semantic",
        dequeue_handler=env.processor,
        middlewares=[TaskWorkQueueMiddleware()],
    )
    msg = env.message(
        generation_trigger="content_write",
        changes={"modified": [env.root_uri + "/content.md"]},
    )
    task = await tracker.create("add_resource", account_id=msg.account_id, user_id=msg.user_id)
    await tracker.start(task.task_id, account_id=msg.account_id, user_id=msg.user_id)
    with bind_task_context(task.task_id, msg.account_id, msg.user_id):
        await queue.enqueue(msg.to_dict())
    payload = backend.write.call_args.args[1].decode()
    backend.read = AsyncMock(
        side_effect=[
            json.dumps({"id": "queue-message", "data": payload}).encode(),
            json.dumps({"pending": 0, "processing": 0}).encode(),
        ]
    )
    await queue.dequeue()
    status = await queue.get_status()
    assert status.error_count == 1 and status.processed == 0
    failed = await tracker.get(task.task_id, account_id=msg.account_id, user_id=msg.user_id)
    assert failed.status is TaskStatus.FAILED
    assert "lack byte compare" in failed.error
    assert env.content.read_bytes() == b"original"
    env.constructor.assert_not_called()


async def test_cancel_during_acquisition_releases_completed_lease(env, monkeypatch):
    started = asyncio.Event()
    resume = asyncio.Event()
    original = env.agfs.pathlock_acquire_batch

    async def acquire(*args, **kwargs):
        started.set()
        await resume.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(env.agfs, "pathlock_acquire_batch", acquire)
    task = asyncio.create_task(env.processor.on_dequeue(env.message().to_dict()))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    resume.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    env.constructor.assert_not_called()
    monkeypatch.setattr(env.agfs, "pathlock_acquire_batch", original)
    lease = await env.agfs.pathlock_acquire_tree("/local/account/resources")
    await env.agfs.pathlock_release(lease)


async def test_borrowed_resource_coverage_is_never_released(env):
    lease = await env.agfs.pathlock_acquire_tree("/local/account/resources")
    try:
        result = await env.processor.on_dequeue(env.message().to_dict(), lock=lease)
        assert result.outcome is ProcessOutcome.SUCCESS
        with pytest.raises(LockAcquisitionError):
            await env.agfs.pathlock_acquire_exact("/local/account/resources/contender")
    finally:
        await env.agfs.pathlock_release(lease)
