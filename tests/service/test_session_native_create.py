# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Real native session creation, without pre-seeding directories or files."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.server.routers import sessions as router
from openviking_cli.exceptions import (
    AlreadyExistsError,
    FailedPreconditionError,
    NotFoundError,
    PermissionDeniedError,
)
from openviking_cli.session.user_id import UserIdentifier
from tests.service.test_session_explicit_append import append
from tests.service.test_session_explicit_append import native as native


@pytest.mark.asyncio
async def test_native_create_explicit_null_overrides_inherited_policy_and_appends(
    native, monkeypatch
):
    sessions, fs, ctx = native
    sessions.set_default_user_auto_commit_policy({"message_count_threshold": 1})
    monkeypatch.setattr(
        router,
        "get_service",
        lambda: SimpleNamespace(sessions=sessions, initialize_user_directories=AsyncMock()),
    )
    request = router.CreateSessionRequest.model_validate({"auto_commit_policy": None})
    created = await router.create_session(request, _ctx=ctx)
    sid = created.result["session_id"]
    assert created.result["auto_commit_policy"] is None
    fresh = await sessions.get(sid, ctx)
    assert fresh.meta.auto_commit_policy is None
    assert fresh.messages == []
    await append(sid, ctx)
    persisted = await sessions.get(sid, ctx)
    assert persisted.meta.auto_commit_policy is None
    assert [message.content for message in persisted.messages] == ["explicit captured text"]
    assert persisted.meta.commit_count == 0
    sessions.maybe_schedule_auto_commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_omitted_policy_keeps_inheritance_and_refuses_qualified_append(native):
    sessions, fs, ctx = native
    sessions.set_default_user_auto_commit_policy({"message_count_threshold": 1})
    created = await sessions.create(ctx, "inherited")
    assert created.meta.auto_commit_policy is not None
    with pytest.raises(FailedPreconditionError):
        await append("inherited", ctx)
    assert (await sessions.get("inherited", ctx)).messages == []
    sessions.maybe_schedule_auto_commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_duplicate_create_and_stale_ensure_do_not_clobber(native):
    sessions, fs, ctx = native
    stale = sessions.session(ctx, "no-clobber")
    created = await sessions.create(ctx, "no-clobber", update_auto_commit_policy=True)
    await append("no-clobber", ctx)
    before = {
        filename: await fs.read_file(created.uri + "/" + filename, ctx=ctx)
        for filename in ["messages.jsonl", ".meta.json"]
    }
    with pytest.raises(AlreadyExistsError):
        await sessions.create(ctx, "no-clobber", auto_commit_policy={})
    stale.meta.auto_commit_policy = {"message_count_threshold": 1}
    await stale.ensure_exists()
    for filename, content in before.items():
        assert await fs.read_file(created.uri + "/" + filename, ctx=ctx) == content


@pytest.mark.asyncio
async def test_native_mkdir_failure_releases_mutex_and_retry_creates(native, monkeypatch):
    sessions, fs, ctx = native
    mkdir = fs.mkdir
    failed = False

    async def fail_once(uri, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            assert kwargs["lease_ref"]["lease_ref"]
            raise OSError("injected mkdir failure")
        return await mkdir(uri, **kwargs)

    monkeypatch.setattr(fs, "mkdir", fail_once)
    with pytest.raises(OSError, match="injected mkdir failure"):
        await sessions.create(ctx, "retry", update_auto_commit_policy=True)
    assert not await sessions.session(ctx, "retry").exists()
    created = await sessions.create(ctx, "retry", update_auto_commit_policy=True)
    await append(created.session_id, ctx)
    assert len((await sessions.get("retry", ctx)).messages) == 1


@pytest.mark.asyncio
async def test_native_new_create_same_user_and_foreign_owner_isolation(native):
    sessions, fs, ctx = native
    created = await sessions.create(ctx, "shared", update_auto_commit_policy=True)
    sibling_ctx = replace(ctx, actor_peer_id="codex")
    await append("shared", sibling_ctx)
    for user in [UserIdentifier(ctx.account_id, "other"), UserIdentifier("other-account", "igor")]:
        foreign_ctx = RequestContext(user=user, role=Role.USER)
        with pytest.raises(NotFoundError):
            await sessions.get("shared", foreign_ctx)
        own = await sessions.create(foreign_ctx, "shared", update_auto_commit_policy=True)
        await append("shared", foreign_ctx)
        assert own.ctx.account_id == user.account_id
        assert own.ctx.user.user_id == user.user_id
        assert len((await sessions.get("shared", foreign_ctx)).messages) == 1
        assert len((await sessions.get("shared", ctx)).messages) == 1
    with pytest.raises(PermissionDeniedError):
        await fs.mkdir(
            created.uri + "/foreign",
            ctx=RequestContext(user=UserIdentifier(ctx.account_id, "other"), role=Role.USER),
        )
    assert not await fs.exists(created.uri + "/foreign", ctx=ctx)


@pytest.mark.asyncio
async def test_native_initialization_mutex_blocks_append_until_meta_is_persisted(
    native, monkeypatch
):
    sessions, fs, ctx = native
    entered, release = asyncio.Event(), asyncio.Event()
    original_write = fs.write_file

    async def pause_meta(uri, content, **kwargs):
        if uri.endswith("/initializing/.meta.json") and not release.is_set():
            entered.set()
            await release.wait()
        return await original_write(uri, content, **kwargs)

    monkeypatch.setattr(fs, "write_file", pause_meta)
    create_task = asyncio.create_task(
        sessions.create(ctx, "initializing", update_auto_commit_policy=True)
    )
    append_task = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        append_task = asyncio.create_task(append("initializing", ctx))
        await asyncio.sleep(0.05)
        assert not append_task.done()
    finally:
        release.set()
        tasks = [task for task in [create_task, append_task] if task is not None]
        await asyncio.wait_for(asyncio.gather(*tasks), 10)
    fresh = await sessions.get("initializing", ctx)
    assert len(fresh.messages) == 1
    assert fresh.meta.message_count == 1
    assert fresh.meta.pending_tokens > 0
    assert fresh.meta.auto_commit_policy is None
