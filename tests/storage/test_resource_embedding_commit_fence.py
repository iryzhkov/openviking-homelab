# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Persisted resource embeddings must revalidate at the real local-store commit."""

import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio

from openviking.core.context import Context, Vectorize
from openviking.server.api_keys.legacy import ACCOUNTS_PATH
from openviking.storage.collection_schemas import TextEmbeddingHandler
from openviking.storage.errors import LockAcquisitionError
from openviking.storage.queuefs.embedding_msg import EmbeddingMsg
from openviking.storage.queuefs.embedding_msg_converter import EmbeddingMsgConverter
from openviking.storage.queuefs.embedding_queue import EmbeddingQueue
from openviking.storage.queuefs.named_queue import NamedQueue
from openviking.storage.queuefs.process_result import ProcessOutcome
from openviking.storage.vector_ids import vector_record_id
from openviking.utils.content_hash import content_md5
from tests.storage.test_transfer_merge_binding import (
    binding_fs as binding_fs,
)
from tests.storage.test_transfer_merge_binding import (
    indexed_fs as indexed_fs,
)
from tests.storage.test_transfer_merge_binding import (
    root_ctx,
)


@pytest_asyncio.fixture
async def env(indexed_fs, monkeypatch, tmp_path):
    fs, db = indexed_fs
    ctx = root_ctx()
    uri = "viking://resources/nested/document.md"
    await fs.write_file_bytes(uri, b"original bytes", ctx=ctx)
    record_id = vector_record_id(ctx.account_id, uri, 2)
    await db.upsert(
        {
            "id": record_id,
            "uri": uri,
            "level": 2,
            "account_id": ctx.account_id,
            "vector": [0.1] * 4,
            "abstract": "protected existing record",
            "search_tags": ["owner=keep"],
        },
        ctx=ctx,
    )
    monkeypatch.setattr("openviking.storage.viking_fs._base._instance", fs)
    monkeypatch.setattr(
        "openviking.storage.collection_schemas.embed_compat",
        AsyncMock(return_value=SimpleNamespace(dense_vector=[0.2] * 4, sparse_vector=None)),
    )
    handler = TextEmbeddingHandler(db, SimpleNamespace(bind=lambda _: object()))
    persisted = tmp_path / "persisted-message.json"
    saved = []
    original_enqueue = NamedQueue.enqueue

    async def save(queue, data):
        del queue
        persisted.write_text(json.dumps(data))
        saved.append(json.loads(persisted.read_text()))
        return "persisted"

    monkeypatch.setattr(NamedQueue, "enqueue", save)
    queue = EmbeddingQueue(fs.agfs, "/queue", "Embedding")

    async def produce(*, hold=False, expected=b"original bytes", level=2, coverage_uri="viking://resources"):
        lease = await fs._async_agfs.pathlock_acquire_tree(
            fs._uri_to_path(coverage_uri, ctx=ctx)
        )
        identity = await fs._capture_resource_identity(ctx)
        try:
            # The pre-fix baseline has no resource producer binding. Its actual
            # enqueue and final commit paths still run, demonstrating stale replay.
            try:
                from openviking.storage.queuefs.resource_embedding_fence import (
                    ResourceEmbeddingProducer,
                    bind_resource_embedding_producer,
                )
            except ImportError:
                binding = nullcontext()
            else:
                binding = bind_resource_embedding_producer(
                    ResourceEmbeddingProducer(ctx, identity, lease)
                )
            context = Context(
                uri=uri,
                abstract="old generated summary",
                user=ctx.user,
                account_id=ctx.account_id,
                context_type="resource",
                level=level,
                is_leaf=level == 2,
                md5=content_md5(expected),
            )
            context.set_vectorize(Vectorize(text="old embedding input"))
            msg = EmbeddingMsgConverter.from_context(context)
            with binding:
                await queue.enqueue(msg)
            data = json.loads(persisted.read_text())
            # Reconstruct the envelope exactly as a different worker would.
            restored = EmbeddingMsg.from_dict(data)
            assert restored.to_dict() == data
            if hold:
                return {"data": restored.to_json()}, lease
            return {"data": restored.to_json()}
        finally:
            if not hold:
                await fs._async_agfs.pathlock_release(lease)

    async def registry_change(change):
        raw = await fs._async_agfs.cat(ACCOUNTS_PATH, fs_ctx={"bypass_cache": "true"})
        data = json.loads(raw)
        if change == "absent":
            del data["accounts"][ctx.account_id]
        elif change == "deleting":
            data["accounts"][ctx.account_id]["deletion"] = {"task_id": "deletion"}
        else:
            data["accounts"][ctx.account_id]["creation_id"] = str(uuid4())
        await fs._async_agfs.write(ACCOUNTS_PATH, json.dumps(data).encode())

    return SimpleNamespace(
        fs=fs,
        db=db,
        ctx=ctx,
        uri=uri,
        record_id=record_id,
        handler=handler,
        produce=produce,
        registry_change=registry_change,
        persisted=persisted,
        queue=queue,
        saved=saved,
        original_enqueue=original_enqueue,
    )


@pytest.mark.parametrize(
    "change", ["content", "delete", "absent", "deleting", "recreated", "metadata"]
)
async def test_persisted_stale_resource_never_overwrites_current_store(env, change):
    data = await env.produce()
    if change == "content":
        await env.fs.write_file_bytes(env.uri, b"replacement bytes", ctx=env.ctx)
    elif change == "delete":
        await env.fs._async_agfs.rm(env.fs._uri_to_path(env.uri, ctx=env.ctx))
    elif change == "metadata":
        await env.db.update_search_tags(
            env.uri, ["owner=new-policy"], mode="replace", levels=[2], ctx=env.ctx
        )
    else:
        await env.registry_change(change)
    before = await env.db.get_strict([env.record_id], ctx=env.ctx)
    result = await env.handler.on_dequeue(data)
    assert result.outcome is ProcessOutcome.FAILED
    assert await env.db.get_strict([env.record_id], ctx=env.ctx) == before


async def test_normal_persisted_resource_commits_under_root_coverage(env, monkeypatch):
    data = await env.produce()
    original = env.db.upsert
    entered = []

    async def guarded(record, **kwargs):
        with pytest.raises(LockAcquisitionError):
            await env.fs._async_agfs.pathlock_acquire_exact(
                env.fs._uri_to_path("viking://resources/foreign", ctx=env.ctx)
            )
        entered.append(True)
        return await original(record, **kwargs)

    monkeypatch.setattr(env.db, "upsert", guarded)
    result = await env.handler.on_dequeue(data)
    assert result.outcome is ProcessOutcome.SUCCESS
    assert entered == [True]
    stored = await env.db.get_strict([env.record_id], ctx=env.ctx)
    assert stored[0]["abstract"] == "old generated summary"
    assert stored[0]["search_tags"] == ["owner=keep"]


async def test_identity_is_rechecked_after_final_acquisition(env, monkeypatch):
    data = await env.produce()
    original = env.fs._async_agfs.pathlock_acquire_batch

    async def acquire(*args, **kwargs):
        lease = await original(*args, **kwargs)
        await env.registry_change("recreated")
        return lease

    monkeypatch.setattr(env.fs._async_agfs, "pathlock_acquire_batch", acquire)
    before = await env.db.get_strict([env.record_id], ctx=env.ctx)
    assert (await env.handler.on_dequeue(data)).outcome is ProcessOutcome.FAILED
    assert await env.db.get_strict([env.record_id], ctx=env.ctx) == before


async def test_unqualified_or_stale_producer_cannot_persist_fresh_proof(env):
    from openviking.storage.queuefs.resource_embedding_fence import ResourceEmbeddingConflict

    legacy = EmbeddingMsg.for_embed(
        message="old", context_data={"uri": env.uri, "account_id": env.ctx.account_id, "level": 2}
    )
    with pytest.raises(ResourceEmbeddingConflict, match="no owned input scope"):
        await env.queue.enqueue(legacy)
    assert not env.persisted.exists()
    await env.fs.write_file_bytes(env.uri, b"new bytes", ctx=env.ctx)
    with pytest.raises(ResourceEmbeddingConflict, match="matching byte condition"):
        await env.produce()
    assert not env.persisted.exists()


async def test_foreign_root_lease_and_directory_are_preserved(env):
    data = await env.produce()
    await env.fs.mkdir("viking://resources/foreign-empty", ctx=env.ctx)
    root = env.fs._uri_to_path("viking://resources", ctx=env.ctx)
    owner = await env.fs._async_agfs.pathlock_acquire_tree(root)
    token = await env.fs._async_agfs.cat(root + "/.path.ovlock")
    try:
        assert (await env.handler.on_dequeue(data)).outcome is ProcessOutcome.FAILED
        assert await env.fs._async_agfs.cat(root + "/.path.ovlock") == token
        assert await env.fs.exists("viking://resources/foreign-empty", ctx=env.ctx)
    finally:
        await env.fs._async_agfs.pathlock_release(owner)


async def test_lost_final_ownership_refuses_upsert_without_foreign_token_cleanup(env, monkeypatch):
    data = await env.produce()
    original = env.db.get_strict
    root = env.fs._uri_to_path("viking://resources", ctx=env.ctx)
    foreign = []

    async def get(*args, **kwargs):
        result = await original(*args, **kwargs)
        raw = await env.fs._async_agfs.cat(root + "/.path.ovlock")
        _, timestamp, kind = raw.decode().split(":")
        token = f"foreign-owner:{timestamp}:{kind}".encode()
        await env.fs._async_agfs.write(root + "/.path.ovlock", token)
        foreign.append(token)
        return result

    monkeypatch.setattr(env.db, "get_strict", get)
    result = await env.handler.on_dequeue(data)
    assert result.outcome is ProcessOutcome.FAILED
    assert foreign
    assert await env.fs._async_agfs.cat(root + "/.path.ovlock") == foreign[0]
    monkeypatch.setattr(env.db, "get_strict", original)
    assert (await env.db.get_strict([env.record_id], ctx=env.ctx))[0][
        "abstract"
    ] == "protected existing record"


@pytest.mark.parametrize(
    "uri",
    [
        "viking://user/alice/memories/item",
        "viking://agent/alice/item",
        "viking://session/example/item",
    ],
)
async def test_nonresource_embedding_behavior_does_not_require_resource_proof(env, uri):
    msg = EmbeddingMsg.for_embed(
        message="current",
        context_data={
            "uri": uri,
            "account_id": env.ctx.account_id,
            "owner_user_id": "default",
            "level": 2,
            "context_type": "memory",
            "is_leaf": True,
            "abstract": "current",
        },
    )
    await env.queue.enqueue(msg)
    result = await env.handler.on_dequeue({"data": env.persisted.read_text()})
    assert result.outcome is ProcessOutcome.SUCCESS


@pytest.mark.parametrize("action", ["update_fields", "delete"])
async def test_legacy_resource_patch_and_delete_cannot_recreate_or_remove_context(env, action):
    if action == "delete":
        msg = EmbeddingMsg.for_delete(
            record_ids=[env.record_id],
            context_data={
                "uri": env.uri,
                "account_id": env.ctx.account_id,
                "level": 2,
            },
        )
    else:
        msg = EmbeddingMsg.for_update_fields(
            record_id=env.record_id,
            fields={"abstract": "old"},
            initial_fields={
                "uri": env.uri,
                "account_id": env.ctx.account_id,
                "level": 2,
                "vector": [0.1] * 4,
            },
            context_data={"uri": env.uri, "account_id": env.ctx.account_id, "level": 2},
        )
    before = await env.db.get_strict([env.record_id], ctx=env.ctx)
    result = await env.handler.on_dequeue({"data": msg.to_json()})
    assert result.outcome is ProcessOutcome.FAILED
    assert await env.db.get_strict([env.record_id], ctx=env.ctx) == before


async def test_directory_producer_persists_both_levels_and_checks_sidecar_bytes(env, monkeypatch):
    from openviking.storage.queuefs.resource_embedding_fence import (
        ResourceEmbeddingProducer,
        bind_resource_embedding_producer,
    )
    from openviking.utils.embedding_utils import vectorize_directory_meta

    uri = "viking://resources/directory"
    body = "directory body"
    for name in (".abstract.md", ".overview.md"):
        await env.fs.write_file_bytes(f"{uri}/{name}", body.encode(), ctx=env.ctx)
    manager = SimpleNamespace(EMBEDDING="Embedding", get_queue=lambda _: env.queue)
    monkeypatch.setattr("openviking.utils.embedding_utils.get_queue_manager", lambda: manager)
    lease = await env.fs._async_agfs.pathlock_acquire_tree(
        env.fs._uri_to_path("viking://resources", ctx=env.ctx)
    )
    identity = await env.fs._capture_resource_identity(env.ctx)
    try:
        with bind_resource_embedding_producer(ResourceEmbeddingProducer(env.ctx, identity, lease)):
            assert await vectorize_directory_meta(
                uri,
                body,
                body,
                ctx=env.ctx,
                md5s={0: content_md5(body.encode()), 1: content_md5(body.encode())},
            ) == {0, 1}
    finally:
        await env.fs._async_agfs.pathlock_release(lease)
    assert len(env.saved) == 2
    for payload in env.saved:
        assert (
            await env.handler.on_dequeue({"data": json.dumps(payload)})
        ).outcome is ProcessOutcome.SUCCESS
    # A later persisted retry carries the original context revision and cannot
    # overwrite the newly committed record.
    assert (
        await env.handler.on_dequeue({"data": json.dumps(env.saved[0])})
    ).outcome is ProcessOutcome.FAILED


@pytest.mark.parametrize("coverage_uri", ["viking://resources", "viking://resources/nested"])
async def test_live_owner_extension_does_not_deadlock_or_release_producer(env, coverage_uri):
    data, owner = await env.produce(hold=True, coverage_uri=coverage_uri)
    try:
        result = await env.handler.on_dequeue(data)
        assert result.outcome is ProcessOutcome.SUCCESS
        assert await env.fs._async_agfs.pathlock_refresh(owner) == "refreshed"
    finally:
        await env.fs._async_agfs.pathlock_release(owner)


async def test_source_changed_during_embedding_is_rechecked_at_commit(env, monkeypatch):
    data = await env.produce()

    async def embed(*args, **kwargs):
        await env.fs.write_file_bytes(env.uri, b"changed during model call", ctx=env.ctx)
        return SimpleNamespace(dense_vector=[0.2] * 4, sparse_vector=None)

    monkeypatch.setattr("openviking.storage.collection_schemas.embed_compat", embed)
    before = await env.db.get_strict([env.record_id], ctx=env.ctx)
    result = await env.handler.on_dequeue(data)
    assert result.outcome is ProcessOutcome.FAILED
    assert await env.db.get_strict([env.record_id], ctx=env.ctx) == before


async def test_legacy_resource_message_cannot_resurrect_missing_index(env):
    data = await env.produce()
    payload = json.loads(data["data"])
    payload["payload"]["context_data"].pop("_resource_fence", None)
    await env.db.strict_delete([env.record_id], ctx=env.ctx)
    result = await env.handler.on_dequeue({"data": json.dumps(payload)})
    assert result.outcome is ProcessOutcome.FAILED
    assert not await env.db.get_strict([env.record_id], ctx=env.ctx)


async def test_cancelled_vector_write_keeps_lease_until_actual_upsert_settles(env, monkeypatch):
    data = await env.produce()
    original = env.db.upsert
    entered, resume = asyncio.Event(), asyncio.Event()

    async def blocked(record, **kwargs):
        entered.set()
        await resume.wait()
        return await original(record, **kwargs)

    monkeypatch.setattr(env.db, "upsert", blocked)
    task = asyncio.create_task(env.handler.on_dequeue(data))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    with pytest.raises(LockAcquisitionError):
        await env.fs._async_agfs.pathlock_acquire_tree(
            env.fs._uri_to_path("viking://resources", ctx=env.ctx)
        )
    resume.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    lease = await env.fs._async_agfs.pathlock_acquire_tree(
        env.fs._uri_to_path("viking://resources", ctx=env.ctx)
    )
    await env.fs._async_agfs.pathlock_release(lease)


async def test_shared_scheduler_keeps_the_original_resource_producer(env):
    from openviking.storage.queuefs.resource_embedding_fence import (
        ResourceEmbeddingProducer,
        bind_resource_embedding_producer,
    )
    from openviking.storage.queuefs.semantic_executor import SemanticTreeExecutor, SemanticTreeWork

    lease = await env.fs._async_agfs.pathlock_acquire_tree(
        env.fs._uri_to_path("viking://resources", ctx=env.ctx)
    )
    identity = await env.fs._capture_resource_identity(env.ctx)
    try:
        with bind_resource_embedding_producer(ResourceEmbeddingProducer(env.ctx, identity, lease)):
            executor = SemanticTreeExecutor(SimpleNamespace(), "resource", 1, env.ctx)
        context = Context(
            uri=env.uri,
            abstract="fresh",
            user=env.ctx.user,
            context_type="resource",
            level=2,
            is_leaf=True,
            md5=content_md5(b"original bytes"),
        )
        msg = EmbeddingMsgConverter.from_context(context)

        async def enqueue(_):
            await env.queue.enqueue(msg)

        executor._run_work_bound = enqueue
        await executor._run_work_with_context(SemanticTreeWork(kind="file", dir_uri=env.uri))
        assert json.loads(env.persisted.read_text())["payload"]["context_data"]["_resource_fence"]
    finally:
        await env.fs._async_agfs.pathlock_release(lease)


async def test_resource_failure_is_reported_by_actual_queue_task_middleware(env, monkeypatch):
    from openviking.service.task_queue_middleware import TaskWorkQueueMiddleware
    from openviking.service.task_store import PersistentTaskStore
    from openviking.service.task_tracker import TaskStatus, TaskTracker
    from openviking.service.task_work_index import bind_task_context
    from tests.test_task_tracker import _FakeAgfs

    data = await env.produce()
    await env.fs.write_file_bytes(env.uri, b"intervening replacement", ctx=env.ctx)
    tracker = TaskTracker(PersistentTaskStore(_FakeAgfs()))
    monkeypatch.setattr("openviking.service.task_tracker._instance", tracker)
    backend = SimpleNamespace(mkdir=AsyncMock(), write=AsyncMock(return_value="message"))
    monkeypatch.setattr("openviking.storage.queuefs.named_queue.AsyncAGFSClient", lambda _: backend)
    queue = NamedQueue(
        object(),
        "/queue",
        "Embedding",
        dequeue_handler=env.handler,
        middlewares=[TaskWorkQueueMiddleware()],
    )
    owner = {"account_id": env.ctx.account_id, "user_id": env.ctx.user.user_id}
    task = await tracker.create("add_resource", **owner)
    await tracker.start(task.task_id, **owner)
    with bind_task_context(task.task_id, **owner):
        await env.original_enqueue(queue, json.loads(data["data"]))
    raw = backend.write.call_args.args[1].decode()
    backend.read = AsyncMock(
        side_effect=[
            json.dumps({"id": "message", "data": raw}).encode(),
            json.dumps({"pending": 0, "processing": 0}).encode(),
        ]
    )
    await queue.dequeue()
    status = await queue.get_status()
    assert status.error_count == 1 and status.processed == 0
    assert (await tracker.get(task.task_id, **owner)).status is TaskStatus.FAILED

@pytest.mark.parametrize("mutate", [False, True])
async def test_enabled_acl_policy_is_retained_or_refuses_stale_context(env, mutate):
    from openviking.storage.acl import ACL_GRANT_FIELDS, ACL_MODE_FIELD, AclManager, AclMode
    from tests.storage.test_transfer_merge_binding import _EnabledAclConfig

    env.fs.acl_manager = env.db.acl_manager = AclManager(env.db, _EnabledAclConfig())
    await env.fs.set_acl(
        env.uri, [{"principal": f"user:{env.ctx.user.user_id}", "level": "manage"}],
        ctx=env.ctx, acl_mode=AclMode.RESTRICTED,
    )
    data = await env.produce()
    if mutate:
        await env.fs.set_acl(
            env.uri, [{"principal": "user:new-owner", "level": "manage"}],
            ctx=env.ctx, acl_mode=AclMode.RESTRICTED,
        )
    before = (await env.db.get_strict([env.record_id], ctx=env.ctx))[0]
    result = await env.handler.on_dequeue(data)
    assert result.outcome is (ProcessOutcome.FAILED if mutate else ProcessOutcome.SUCCESS)
    after = (await env.db.get_strict([env.record_id], ctx=env.ctx))[0]
    assert all(after.get(field) == before.get(field) for field in [ACL_MODE_FIELD, *ACL_GRANT_FIELDS])
