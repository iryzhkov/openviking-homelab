# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Account teardown uses the existing native lease manager; no live services."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.pyagfs import AGFSNotFoundError, AsyncAGFSClient
from openviking.server.identity import RequestContext, Role
from openviking.service.deletion import DeletionService
from openviking.service.task_tracker import TaskStatus
from openviking.storage.errors import LockAcquisitionError
from openviking.storage.viking_fs._access import _AccessMixin
from openviking_cli.session.user_id import UserIdentifier


def _ctx():
    return RequestContext(user=UserIdentifier("account", "operator"), role=Role.ROOT)


def _deletion(agfs, vectors=None, manager=None):
    fs = SimpleNamespace(
        _async_agfs=agfs,
        vector_store=vectors,
        _pathlock_fs_ctx=lambda ctx, lease: _AccessMixin._pathlock_fs_ctx(
            SimpleNamespace(_ctx_or_default=lambda ctx: ctx), ctx, lease
        ),
    )
    service = SimpleNamespace(
        viking_fs=fs, watch_scheduler=None, runtime_config_manager=None,
        release_account_vector_resources=AsyncMock(),
    )
    return DeletionService(
        service=service, manager=manager, service_loop=asyncio.get_running_loop()
    )


@pytest.fixture
def native(tmp_path):
    import ragfs_python

    client = ragfs_python.RAGFSBindingClient()
    client.mount("localfs", "/local", {"local_dir": str(tmp_path)})
    return AsyncAGFSClient(client), tmp_path


async def test_foreign_live_lease_preserves_files_vectors_and_empty_directory(native):
    agfs, root = native
    (root / "account/resources/foreign-empty").mkdir(parents=True)
    (root / "account/resources/content").write_text("keep")
    lease = await agfs.pathlock_acquire_tree("/local/account/resources")
    vectors = SimpleNamespace(delete_account_data=AsyncMock())
    try:
        with pytest.raises(LockAcquisitionError):
            await asyncio.wait_for(
                _deletion(agfs, vectors)._delete_account_files("account", _ctx()), 2
            )
        assert (root / "account/resources/content").read_text() == "keep"
        assert (root / "account/resources/foreign-empty").is_dir()
        vectors.delete_account_data.assert_not_awaited()
    finally:
        await agfs.pathlock_release(lease)
    await _deletion(agfs, vectors)._delete_account_files("account", _ctx())
    assert not (root / "account").exists()
    vectors.delete_account_data.assert_awaited_once()


async def test_vector_and_file_phases_share_account_coverage(native, monkeypatch):
    agfs, root = native
    (root / "account/resources").mkdir(parents=True)
    (root / "account/resources/content").write_text("old")
    events = []
    original_rm = agfs.rm

    async def assert_foreign_refused():
        with pytest.raises(LockAcquisitionError):
            await agfs.pathlock_acquire_exact("/local/account/resources/contender")

    async def delete_vectors(account_id, *, ctx):
        assert account_id == ctx.account_id == "account"
        await assert_foreign_refused()
        assert (root / "account/resources/content").read_text() == "old"
        events.append("vectors")

    async def remove(path, **kwargs):
        assert path == "/local/account"
        assert kwargs["fs_ctx"]["account_id"] == "account"
        assert kwargs["fs_ctx"]["lease_ref"]
        assert "auto_pathlock" not in kwargs
        await assert_foreign_refused()
        events.append("files")
        return await original_rm(path, **kwargs)

    monkeypatch.setattr(agfs, "rm", remove)
    await _deletion(
        agfs, SimpleNamespace(delete_account_data=delete_vectors)
    )._delete_account_files("account", _ctx())
    assert events == ["vectors", "files"]
    assert not (root / "account").exists()


async def test_stale_descendant_token_does_not_block(native):
    agfs, root = native
    resources = root / "account/resources"
    resources.mkdir(parents=True)
    # Persist an expired token, without a live owner/heartbeat to renew it.
    lease = await agfs.pathlock_acquire_tree("/local/account/resources")
    token_path = resources / ".path.ovlock"
    owner, timestamp, kind = token_path.read_text().split(":")
    assert int(timestamp) > 1 and kind == "T"
    await agfs.pathlock_release(lease)
    token_path.write_text(f"{owner}:1:{kind}")
    started = time.monotonic()
    await asyncio.wait_for(_deletion(agfs)._delete_account_files("account", _ctx()), 2)
    assert time.monotonic() - started < 2
    assert not (root / "account").exists()


@pytest.mark.parametrize("phase", ["vectors", "files", "release"])
async def test_failure_releases_lease_without_unlocked_fallback(phase):
    error = OSError("injected failure")
    lease = {"lease_ref": "owned"}
    agfs = SimpleNamespace(
        pathlock_acquire_tree=AsyncMock(return_value=lease),
        pathlock_release=AsyncMock(side_effect=error if phase == "release" else None),
        rm=AsyncMock(side_effect=error if phase == "files" else None),
        stat=AsyncMock(side_effect=AGFSNotFoundError("missing")),
    )
    vectors = SimpleNamespace(
        delete_account_data=AsyncMock(side_effect=error if phase == "vectors" else None)
    )
    with pytest.raises(OSError, match="injected failure"):
        await _deletion(agfs, vectors)._delete_account_files("account", _ctx())
    agfs.pathlock_release.assert_awaited_once_with(lease)
    assert agfs.rm.await_count == (0 if phase == "vectors" else 1)
    if agfs.rm.await_count:
        assert "auto_pathlock" not in agfs.rm.await_args.kwargs
    agfs.stat.assert_not_awaited()


async def test_missing_own_token_is_tolerated_but_remaining_root_fails():
    lease = {"lease_ref": "owned"}
    agfs = SimpleNamespace(
        pathlock_acquire_tree=AsyncMock(return_value=lease),
        pathlock_release=AsyncMock(side_effect=AGFSNotFoundError("own token removed")),
        rm=AsyncMock(), stat=AsyncMock(return_value={"isDir": True}),
    )
    with pytest.raises(RuntimeError, match="still exists"):
        await _deletion(agfs)._delete_account_files("account", _ctx())
    agfs.rm.assert_awaited_once()
    agfs.stat.assert_awaited_once_with("/local/account")


async def test_busy_task_keeps_fence_and_resubmission_replaces_failed_task(monkeypatch):
    from openviking.service import deletion as module

    fence = {"task_id": "old", "owner_account_id": "system", "owner_user_id": "system"}
    manager = SimpleNamespace(
        get_deletion=AsyncMock(side_effect=lambda *args: dict(fence)),
        begin_deletion=AsyncMock(side_effect=lambda *args, **kwargs: (dict(fence), False)),
        finish_deletion=AsyncMock(),
    )
    async def replace(*args, **kwargs):
        assert kwargs["expected_task_id"] == "old"
        fence.update(
            task_id=kwargs["task_id"],
            owner_account_id=kwargs["owner_account_id"],
            owner_user_id=kwargs["owner_user_id"],
        )
        return dict(fence)
    manager.replace_deletion_task = AsyncMock(side_effect=replace)
    task = SimpleNamespace(status=TaskStatus.PENDING)
    async def fail(*args, **kwargs):
        task.status = TaskStatus.FAILED
    tracker = SimpleNamespace(
        create=AsyncMock(return_value=task), start=AsyncMock(),
        fail=AsyncMock(side_effect=fail), complete=AsyncMock(),
        delete_tasks=AsyncMock(), get=AsyncMock(return_value=task),
    )
    monkeypatch.setattr(module, "get_task_tracker", lambda: tracker)
    agfs = SimpleNamespace(
        pathlock_acquire_tree=AsyncMock(side_effect=LockAcquisitionError("foreign busy")),
        pathlock_release=AsyncMock(), rm=AsyncMock(),
    )
    vectors = SimpleNamespace(delete_account_data=AsyncMock())
    deletion = _deletion(agfs, vectors, manager)
    deletion._cancel_tasks = AsyncMock()
    deletion._enqueue_if_missing = AsyncMock()
    error = await deletion._process({
        "task_id": "old", "account_id": "system", "user_id": "system",
        "target": {"account_id": "account", "user_id": None},
    })
    assert "foreign busy" in error
    assert fence["task_id"] == "old"
    manager.finish_deletion.assert_not_awaited()
    vectors.delete_account_data.assert_not_awaited()
    agfs.rm.assert_not_awaited()
    agfs.pathlock_release.assert_not_awaited()
    result = await deletion.delete("account", actor=_ctx())
    assert result["task_id"] == fence["task_id"] != "old"
    manager.replace_deletion_task.assert_awaited_once()
    deletion._enqueue_if_missing.assert_awaited_once()
