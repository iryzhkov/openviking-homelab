# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Existing-only append with the actual native PathLock and local storage."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from openviking.message import TextPart
from openviking.server.identity import RequestContext, Role
from openviking.server.routers import sessions as router
from openviking.service.session_service import SessionService
from openviking.storage.viking_fs import VikingFS
from openviking.utils.agfs_utils import RagfsBindingConfig, create_agfs_client
from openviking_cli.exceptions import FailedPreconditionError, NotFoundError, PermissionDeniedError
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.agfs_config import AGFSConfig

MODE = "existing_only_no_auto_commit"
SPEC = [{"role": "user", "parts": [TextPart(text="explicit captured text")]}]


@pytest.fixture
def native(tmp_path, monkeypatch):
    agfs = create_agfs_client(
        RagfsBindingConfig(agfs=AGFSConfig(path=str(tmp_path / "native"), backend="local"))
    )
    fs = VikingFS(agfs=agfs)
    sessions = SessionService(viking_fs=fs)
    sessions.maybe_schedule_auto_commit = AsyncMock(return_value=False)
    monkeypatch.setattr(router, "get_service", lambda: SimpleNamespace(sessions=sessions))
    ctx = RequestContext(user=UserIdentifier("append-account", "igor"), role=Role.USER)
    return sessions, fs, ctx


async def provision(sessions, ctx, sid, *, enabled=False):
    # Seed an existing native session directly. ensure_exists currently acquires
    # its directory mutex then mkdir without passing the lease to native storage.
    # That unrelated creation bug is outside this existing-only append contract.
    session = sessions.session(ctx, sid)
    await session._viking_fs.mkdir(session.uri, exist_ok=True, ctx=ctx)
    await session._viking_fs.write_file(session.uri + "/messages.jsonl", b"", ctx=ctx)
    session.meta.auto_commit_policy = {} if enabled else None
    await session._save_meta()
    return session


async def append(sid, ctx, mode=MODE):
    return await router.add_message(
        router.AddMessageRequest(role="user", content="explicit captured text", append_mode=mode),
        session_id=sid,
        _ctx=ctx,
    )


@pytest.mark.asyncio
async def test_missing_refuses_without_creation(native):
    sessions, fs, ctx = native
    with pytest.raises(NotFoundError):
        await append("missing", ctx)
    assert not await sessions.session(ctx, "missing").exists()
    sessions.maybe_schedule_auto_commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_persists_and_same_user_sibling_allowed(native):
    sessions, fs, ctx = native
    sessions.set_default_user_auto_commit_policy({"message_count_threshold": 1})
    original = await provision(sessions, ctx, "sibling")
    response = await append("sibling", replace(ctx, actor_peer_id="codex"))
    assert response.result["append_mode"] == MODE
    assert response.result["auto_commit_scheduled"] is False
    fresh = await sessions.get("sibling", ctx)
    assert fresh.meta.auto_commit_policy is None
    assert [m.content for m in fresh.messages] == ["explicit captured text"]
    assert fresh.meta.commit_count == 0
    assert original.uri == fresh.uri
    sessions.maybe_schedule_auto_commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("race", ["enable", "delete"])
async def test_stale_open_rechecks_under_native_lock_before_effects(native, monkeypatch, race):
    sessions, fs, ctx = native
    await provision(sessions, ctx, "raced")
    stale = await sessions.open("raced", ctx)
    original_lock = stale._state_lock
    prepared = AsyncMock(wraps=stale._prepare_appended_messages)
    monkeypatch.setattr(stale, "_prepare_appended_messages", prepared)

    @asynccontextmanager
    async def raced_lock(*names, **kwargs):
        if race == "enable":
            await sessions.update_config("raced", ctx, auto_commit_policy={})
        else:
            await sessions.delete("raced", ctx)
            assert not await stale.exists(), (
                "native delete must remove the session before lock acquisition"
            )
        async with original_lock(*names, **kwargs) as lease:
            yield lease

    monkeypatch.setattr(stale, "_state_lock", raced_lock)
    with pytest.raises(FailedPreconditionError if race == "enable" else NotFoundError):
        await stale.add_messages_async(SPEC, require_existing_no_auto_commit=True)
    prepared.assert_not_awaited()
    if race == "enable":
        fresh = await sessions.get("raced", ctx)
        assert not fresh.messages
        assert fresh.meta.auto_commit_policy is not None
    else:
        assert not await sessions.session(ctx, "raced").exists()


@pytest.mark.asyncio
async def test_policy_update_waits_for_qualified_write_lock(native, monkeypatch):
    sessions, fs, ctx = native
    session = await provision(sessions, ctx, "serialized")
    entered, release = asyncio.Event(), asyncio.Event()
    prepare = session._prepare_appended_messages

    async def paused(specs):
        entered.set()
        await release.wait()
        return await prepare(specs)

    monkeypatch.setattr(session, "_prepare_appended_messages", paused)
    write = asyncio.create_task(
        session.add_messages_async(SPEC, require_existing_no_auto_commit=True)
    )
    await asyncio.wait_for(entered.wait(), 5)
    update = asyncio.create_task(sessions.update_config("serialized", ctx, auto_commit_policy={}))
    try:
        await asyncio.sleep(0.05)
        assert not update.done()
    finally:
        release.set()
    await asyncio.wait_for(asyncio.gather(write, update), 10)
    fresh = await sessions.get("serialized", ctx)
    assert len(fresh.messages) == 1
    assert fresh.meta.auto_commit_policy is not None


@pytest.mark.asyncio
async def test_foreign_user_account_and_explicit_uri_denied(native):
    sessions, fs, ctx = native
    owned = await provision(sessions, ctx, "owned")
    for user in [UserIdentifier("append-account", "foreign"), UserIdentifier("other", "igor")]:
        foreign = RequestContext(user=user, role=Role.USER)
        with pytest.raises(NotFoundError):
            await append("owned", foreign)
        assert not await sessions.session(foreign, "owned").exists()
    with pytest.raises(PermissionDeniedError):
        await fs.read_file(
            owned.uri + "/messages.jsonl",
            ctx=RequestContext(user=UserIdentifier("append-account", "foreign"), role=Role.USER),
        )
    assert not (await sessions.get("owned", ctx)).messages


@pytest.mark.asyncio
async def test_enabled_policy_refuses_before_privacy_or_externalization(native, monkeypatch):
    sessions, fs, ctx = native
    session = await provision(sessions, ctx, "enabled", enabled=True)
    prepare = AsyncMock()
    monkeypatch.setattr(session, "_prepare_appended_messages", prepare)
    with pytest.raises(FailedPreconditionError):
        await session.add_messages_async(SPEC, require_existing_no_auto_commit=True)
    prepare.assert_not_awaited()
    assert not (await sessions.get("enabled", ctx)).messages


@pytest.mark.asyncio
async def test_legacy_single_and_batch_still_schedule(native):
    sessions, fs, ctx = native
    await provision(sessions, ctx, "legacy")
    await provision(sessions, ctx, "batch")
    await append("legacy", ctx, None)
    sessions.maybe_schedule_auto_commit.assert_awaited_once()
    sessions.maybe_schedule_auto_commit.reset_mock()
    result = await router.batch_add_messages(
        router.BatchAddMessageRequest(
            messages=[router.AddMessageRequest(role="assistant", content="batch")]
        ),
        session_id="batch",
        _ctx=ctx,
    )
    assert result.result["added"] == 1
    assert (await sessions.get("legacy", ctx)).messages[0].content == "explicit captured text"
    assert (await sessions.get("batch", ctx)).messages[0].content == "batch"
    sessions.maybe_schedule_auto_commit.assert_awaited_once()


def test_closed_mode_and_batch_does_not_silently_ignore_mode():
    for mode in ["legacy", "", True, 1]:
        with pytest.raises(ValidationError):
            router.AddMessageRequest(role="user", content="text", append_mode=mode)
    with pytest.raises(ValidationError):
        router.BatchAddMessageRequest(
            messages=[router.AddMessageRequest(role="user", content="text", append_mode=MODE)]
        )
