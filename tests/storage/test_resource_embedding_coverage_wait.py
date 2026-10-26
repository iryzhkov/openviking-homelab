# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Embedding fences wait a bounded time for resources-root coverage.

The wait happens in the native lock manager. Every original condition is
still rechecked after the acquisition, and a holder beyond the bound still
fails closed without touching its token.
"""

import asyncio

import pytest

from openviking.storage.queuefs.process_result import ProcessOutcome
from tests.storage.test_resource_embedding_commit_fence import env as env
from tests.storage.test_transfer_merge_binding import binding_fs as binding_fs
from tests.storage.test_transfer_merge_binding import indexed_fs as indexed_fs


def _root(env):
    return env.fs._uri_to_path("viking://resources", ctx=env.ctx)


async def _wait_until_blocked(task):
    await asyncio.sleep(0.4)
    assert not task.done()


async def test_competing_fence_waits_then_commits(env):
    # Observed in the real ingest: a fence whose original owner was handed
    # off acquires root afresh while a sibling fence extending the live
    # semantic owner holds root. Extensions of one live owner do not block
    # each other; a fresh acquisition must wait for the sibling.
    data = await env.produce(coverage_uri="viking://resources/nested")
    owner = await env.fs._async_agfs.pathlock_acquire_tree(
        env.fs._uri_to_path("viking://resources/nested", ctx=env.ctx)
    )
    try:
        identity = await env.fs._capture_resource_identity(env.ctx)
        sibling = await env.fs._acquire_resource_coverage(env.ctx, identity, owner_lease_ref=owner)
        task = asyncio.create_task(env.handler.on_dequeue(data))
        try:
            await _wait_until_blocked(task)
        finally:
            # The sibling fence and then the semantic owner finish within the bound.
            await env.fs._async_agfs.pathlock_release(sibling)
            await env.fs._async_agfs.pathlock_release(owner)
            owner = None
        result = await task
        assert result.outcome is ProcessOutcome.SUCCESS, result
        stored = await env.db.get_strict([env.record_id], ctx=env.ctx)
        assert stored[0]["abstract"] == "old generated summary"
    finally:
        if owner is not None:
            await env.fs._async_agfs.pathlock_release(owner)


async def test_foreign_root_held_beyond_bound_fails_closed(env, monkeypatch):
    monkeypatch.setattr(
        "openviking.storage.queuefs.resource_embedding_fence.RESOURCE_EMBEDDING_COVERAGE_WAIT_SECS",
        0.5,
    )
    data = await env.produce()
    foreign = await env.fs._async_agfs.pathlock_acquire_tree(_root(env))
    token = await env.fs._async_agfs.cat(_root(env) + "/.path.ovlock")
    try:
        before = await env.db.get_strict([env.record_id], ctx=env.ctx)
        started = asyncio.get_running_loop().time()
        result = await env.handler.on_dequeue(data)
        waited = asyncio.get_running_loop().time() - started
        assert result.outcome is ProcessOutcome.FAILED
        assert waited >= 0.4
        assert await env.db.get_strict([env.record_id], ctx=env.ctx) == before
        assert await env.fs._async_agfs.cat(_root(env) + "/.path.ovlock") == token
    finally:
        await env.fs._async_agfs.pathlock_release(foreign)


# The native lock manager refuses source writes while root is held, so a
# concurrent source change during the wait is covered by the existing
# after-acquisition byte recheck tests instead.
@pytest.mark.parametrize("change", ["recreated", "deleting", "absent"])
async def test_change_while_waiting_refuses_after_acquisition(env, change):
    data = await env.produce()
    foreign = await env.fs._async_agfs.pathlock_acquire_tree(_root(env))
    before = await env.db.get_strict([env.record_id], ctx=env.ctx)
    task = asyncio.create_task(env.handler.on_dequeue(data))
    try:
        await _wait_until_blocked(task)
        await env.registry_change(change)
    finally:
        await env.fs._async_agfs.pathlock_release(foreign)
    result = await task
    assert result.outcome is ProcessOutcome.FAILED
    assert await env.db.get_strict([env.record_id], ctx=env.ctx) == before
