# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Resource patch/delete index actions bind original planning conditions.

Conditions are captured while the original request owns its resource lease and
authoritative account identity, carried through the persisted queue envelope,
and rechecked by the consumer under final resources-root coverage.
"""

import asyncio
import json

import pytest

from openviking.storage.context_update_plan import DirectIndexAction
from openviking.storage.errors import LockAcquisitionError
from openviking.storage.index_action import FieldPatch, IndexAction
from openviking.storage.queuefs.embedding_msg import EmbeddingMsg
from openviking.storage.queuefs.process_result import ProcessOutcome
from openviking.storage.queuefs.resource_embedding_fence import (
    ResourceEmbeddingConflict,
    ResourceEmbeddingProducer,
    bind_resource_embedding_producer,
    bind_resource_index_conditions,
    capture_resource_index_conditions,
)
from tests.storage.test_resource_embedding_commit_fence import env as env
from tests.storage.test_transfer_merge_binding import (
    binding_fs as binding_fs,
)
from tests.storage.test_transfer_merge_binding import (
    indexed_fs as indexed_fs,
)

NEW_TAGS = ["owner=new-policy"]


def _patch(tags=NEW_TAGS):
    return FieldPatch({"search_tags": list(tags)}, {"search_tags": "replace"})


def _action(env, kind, patch=None):
    if kind == "delete":
        return DirectIndexAction(IndexAction.DELETE, env.uri, 2, env.record_id)
    return DirectIndexAction(
        IndexAction.UPDATE_FIELDS, env.uri, 2, env.record_id, field_patch=patch or _patch()
    )


def _message(env, action):
    # Mirrors ResourceProcessor._enqueue_index_actions_bound message shape.
    data = {
        "uri": action.uri,
        "level": action.level,
        "account_id": env.ctx.account_id,
        "owner_user_id": env.ctx.user.user_id,
    }
    if action.action is IndexAction.DELETE:
        return EmbeddingMsg.for_delete(record_ids=[action.record_id], context_data=data)
    return EmbeddingMsg.for_update_fields(
        record_id=action.record_id, field_patch=action.field_patch, context_data=data
    )


async def _root_lease(env):
    return await env.fs._async_agfs.pathlock_acquire_tree(
        env.fs._uri_to_path("viking://resources", ctx=env.ctx)
    )


async def _issue(env, kind, *, message_patch=None):
    """Capture under the original owner, enqueue, persist, then release."""
    lease = await _root_lease(env)
    try:
        identity = await env.fs._capture_resource_identity(env.ctx)
        action = _action(env, kind)
        conditions = await capture_resource_index_conditions([action], env.ctx, identity, lease)
        sent = _action(env, kind, message_patch) if message_patch else action
        with bind_resource_index_conditions(conditions, lease):
            await env.queue.enqueue(_message(env, sent))
        data = json.loads(env.persisted.read_text())
        restored = EmbeddingMsg.from_dict(data)
        assert restored.to_dict() == data
        proof = restored.context_data["_resource_fence"]
        return {"data": restored.to_json()}, proof
    finally:
        await env.fs._async_agfs.pathlock_release(lease)


@pytest.mark.parametrize("kind", ["delete", "update_fields"])
async def test_captured_index_action_commits_under_root_coverage(env, monkeypatch, kind):
    data, _ = await _issue(env, kind)
    entered = []
    name = "strict_delete" if kind == "delete" else "get_strict"
    original = getattr(env.db, name)

    async def guarded(*args, **kwargs):
        with pytest.raises(LockAcquisitionError):
            await env.fs._async_agfs.pathlock_acquire_exact(
                env.fs._uri_to_path("viking://resources/foreign", ctx=env.ctx)
            )
        entered.append(True)
        return await original(*args, **kwargs)

    monkeypatch.setattr(env.db, name, guarded)
    result = await env.handler.on_dequeue(data)
    assert result.outcome is ProcessOutcome.SUCCESS, result
    assert entered
    monkeypatch.setattr(env.db, name, original)
    stored = await env.db.get_strict([env.record_id], ctx=env.ctx)
    if kind == "delete":
        assert stored == []
    else:
        assert stored[0]["search_tags"] == NEW_TAGS
        assert stored[0]["abstract"] == "protected existing record"
        assert stored[0]["vector"] == pytest.approx([0.1] * 4)


@pytest.mark.parametrize("kind", ["delete", "update_fields"])
@pytest.mark.parametrize("change", ["metadata", "content", "absent", "deleting", "recreated"])
async def test_stale_index_action_refuses_without_mutation(env, kind, change):
    data, proof = await _issue(env, kind)
    assert proof["creation_identity"]
    assert proof["action"] == IndexAction(kind).value
    if change == "metadata":
        await env.db.update_search_tags(
            env.uri, ["owner=intervening"], mode="replace", levels=[2], ctx=env.ctx
        )
    elif change == "content":
        await env.fs.write_file_bytes(env.uri, b"replacement bytes", ctx=env.ctx)
    else:
        await env.registry_change(change)
    before = await env.db.get_strict([env.record_id], ctx=env.ctx)
    result = await env.handler.on_dequeue(data)
    assert result.outcome is ProcessOutcome.FAILED
    assert await env.db.get_strict([env.record_id], ctx=env.ctx) == before


async def test_conditions_cannot_be_inherited_by_another_owner(env):
    lease = await _root_lease(env)
    try:
        identity = await env.fs._capture_resource_identity(env.ctx)
        action = _action(env, "delete")
        conditions = await capture_resource_index_conditions([action], env.ctx, identity, lease)
    finally:
        await env.fs._async_agfs.pathlock_release(lease)
    later = await _root_lease(env)
    try:
        with bind_resource_index_conditions(conditions, later):
            with pytest.raises(ResourceEmbeddingConflict, match="another owner"):
                await env.queue.enqueue(_message(env, action))
        with bind_resource_index_conditions(conditions, None):
            with pytest.raises(ResourceEmbeddingConflict, match="another owner"):
                await env.queue.enqueue(_message(env, action))
    finally:
        await env.fs._async_agfs.pathlock_release(later)
    assert not env.persisted.exists()


@pytest.mark.parametrize("kind", ["delete", "update_fields"])
async def test_uncaptured_or_processing_time_producers_cannot_issue(env, kind):
    msg = _message(env, _action(env, kind))
    with pytest.raises(ResourceEmbeddingConflict, match="no owned input scope"):
        await env.queue.enqueue(msg)
    with bind_resource_index_conditions({}, {"token": "x"}):
        with pytest.raises(ResourceEmbeddingConflict, match="original captured conditions"):
            await env.queue.enqueue(_message(env, _action(env, kind)))
    lease = await _root_lease(env)
    try:
        identity = await env.fs._capture_resource_identity(env.ctx)
        # Semantic processing admission does not prove the original request lifetime.
        with bind_resource_embedding_producer(ResourceEmbeddingProducer(env.ctx, identity, lease)):
            with pytest.raises(ResourceEmbeddingConflict, match="original admission"):
                await env.queue.enqueue(_message(env, _action(env, kind)))
    finally:
        await env.fs._async_agfs.pathlock_release(lease)
    assert not env.persisted.exists()


async def test_patch_cannot_capture_missing_context(env):
    await env.db.strict_delete([env.record_id], ctx=env.ctx)
    lease = await _root_lease(env)
    try:
        identity = await env.fs._capture_resource_identity(env.ctx)
        with pytest.raises(ResourceEmbeddingConflict, match="missing context"):
            await capture_resource_index_conditions(
                [_action(env, "update_fields")], env.ctx, identity, lease
            )
    finally:
        await env.fs._async_agfs.pathlock_release(lease)
    assert await env.db.get_strict([env.record_id], ctx=env.ctx) == []


async def test_patch_intent_cannot_change_after_capture(env):
    with pytest.raises(ResourceEmbeddingConflict, match="intent changed"):
        await _issue(env, "update_fields", message_patch=_patch(["owner=other"]))
    assert not env.persisted.exists()


async def test_cancelled_delete_keeps_coverage_until_actual_delete_settles(env, monkeypatch):
    data, _ = await _issue(env, "delete")
    original = env.db.strict_delete
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(env.db, "strict_delete", blocked)
    task = asyncio.create_task(env.handler.on_dequeue(data))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    with pytest.raises(LockAcquisitionError):
        await env.fs._async_agfs.pathlock_acquire_exact(
            env.fs._uri_to_path("viking://resources/foreign", ctx=env.ctx)
        )
    release.set()
    try:
        await task
    except asyncio.CancelledError:
        pass
    monkeypatch.setattr(env.db, "strict_delete", original)
    assert await env.db.get_strict([env.record_id], ctx=env.ctx) == []
    after = await _root_lease(env)
    await env.fs._async_agfs.pathlock_release(after)


async def test_nonresource_delete_needs_no_resource_proof(env):
    msg = EmbeddingMsg.for_delete(
        record_ids=["memory-a", "memory-b"],
        context_data={
            "uri": "viking://user/alice/memories/item",
            "account_id": env.ctx.account_id,
            "owner_user_id": "default",
        },
    )
    await env.queue.enqueue(msg)
    data = json.loads(env.persisted.read_text())
    assert "_resource_fence" not in json.dumps(data)
    result = await env.handler.on_dequeue({"data": env.persisted.read_text()})
    assert result.outcome is ProcessOutcome.SUCCESS
