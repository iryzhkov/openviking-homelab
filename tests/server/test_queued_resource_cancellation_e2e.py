# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Cancel a queued resource ingest while its real native write is blocked.

ResourceService.add_resource(wait=False) persists an AddResource message.
The real queue middleware registers Task Work and the real consumer runs
ResourceProcessor on a queue worker thread. A call-through proxy around the
AsyncAGFSClient's native binding client blocks the first resource content
write inside its worker thread, then always calls the native method. Models
are synthetic; filesystem, leases, Task store, queues and vectors are real.
"""

import asyncio
import threading
import time

import pytest

from openviking.service.task_domain import TaskStatus
from openviking.service.task_tracker import get_task_tracker
from openviking_cli.exceptions import ConflictError
from tests.server.conftest import sample_markdown_file as sample_markdown_file
from tests.server.conftest import service as service
from tests.server.conftest import temp_dir as temp_dir
from tests.server.test_real_resource_producer_e2e import _records, _root_ctx
from tests.server.test_real_resource_producer_e2e import admitted_service as admitted_service

TARGET = "viking://resources/sample"


class _BlockingNativeClient:
    """Delegate every call to the native client; block one resource write."""

    def __init__(self, native, prefix):
        self._native = native
        self._prefix = prefix
        self.entered = threading.Event()
        self.release = threading.Event()
        self.blocked_path = None
        self.blocked_thread = None
        self.writes = []

    def __getattr__(self, name):
        return getattr(self._native, name)

    def write(self, path, *args, **kwargs):
        resource_write = str(path).startswith(self._prefix) and not str(path).endswith(
            ".path.ovlock"
        )
        if resource_write and self.blocked_path is None:
            self.blocked_path = str(path)
            self.blocked_thread = threading.current_thread().name
            self.entered.set()
            assert self.release.wait(60), "test never released the blocked write"
        result = self._native.write(path, *args, **kwargs)
        if resource_write:
            self.writes.append((time.monotonic(), str(path), threading.current_thread().name))
        return result


async def _wait_until(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(0.05)
    return False


async def test_cancel_blocked_queued_ingest_gates_until_work_settles(
    admitted_service, sample_markdown_file
):
    service = admitted_service
    ctx = _root_ctx()
    fs = service.viking_fs
    agfs = fs._async_agfs
    prefix = fs._uri_to_path("viking://resources", ctx=ctx) + "/"
    proxy = _BlockingNativeClient(agfs._client, prefix)
    agfs._client = proxy
    tracker = get_task_tracker()
    try:
        result = await service.resources.add_resource(
            path=str(sample_markdown_file), ctx=ctx, to=TARGET, reason="r2", wait=False
        )
        task_id = result["task_id"]
        main_thread = threading.current_thread().name
        assert await asyncio.to_thread(proxy.entered.wait, 30), result
        assert proxy.blocked_path.startswith(prefix)
        # Blocked inside the native call on a worker thread, not the test loop.
        assert proxy.blocked_thread and proxy.blocked_thread != main_thread

        # The write is blocked inside the native call of a queue worker thread.
        task = await tracker.get(task_id, account_id=ctx.account_id, user_id=ctx.user.user_id)
        assert task is not None and task.is_active(), task
        assert await tracker._store.has_open_work(
            task_id, exclude_work_id=None, account_id=ctx.account_id, user_id=ctx.user.user_id
        )
        manager = getattr(service.fs, "_get_watch_manager", lambda: None)()
        if manager is not None:
            assert not await manager.get_all_tasks(ctx.account_id, ctx.user.user_id, "root")

        cancelled = await tracker.cancel(
            task_id, account_id=ctx.account_id, user_id=ctx.user.user_id
        )
        assert cancelled is not None and cancelled.status is TaskStatus.CANCELLING, cancelled
        # Still cancelling while the registered Work and native write are alive.
        await asyncio.sleep(0.3)
        task = await tracker.get(task_id, account_id=ctx.account_id, user_id=ctx.user.user_id)
        assert task.status is TaskStatus.CANCELLING, task
        assert not proxy.writes
        with pytest.raises(ConflictError):
            await service.fs.prune_resource_index(TARGET, ctx)
        with pytest.raises(ConflictError):
            await service.fs.remove_resource_only(TARGET, ctx, recursive=True)

        proxy.release.set()
        terminal = await tracker.wait(
            task_id, account_id=ctx.account_id, user_id=ctx.user.user_id, timeout=60
        )
        assert terminal.status in {TaskStatus.CANCELLED, TaskStatus.FAILED}, terminal
        assert not await tracker._store.has_open_work(
            task_id, exclude_work_id=None, account_id=ctx.account_id, user_id=ctx.user.user_id
        )
        terminal_at = time.monotonic()
        records_at_terminal = await _records(service, ctx)
        writes_at_terminal = len(proxy.writes)

        # Drain anything still queued, then prove no later resource effects.
        await asyncio.sleep(2.0)
        late = [write for write in proxy.writes[writes_at_terminal:] if write[0] > terminal_at]
        assert late == [], late
        resource_records = lambda records: sorted(  # noqa: E731
            (r.get("uri"), r.get("level"), r.get("abstract"))
            for r in records
            if str(r.get("uri", "")).startswith(TARGET)
        )
        assert resource_records(await _records(service, ctx)) == resource_records(
            records_at_terminal
        )
        assert all(write[2] != main_thread for write in proxy.writes)

        print(
            "R2_EVIDENCE",
            {
                "task_id": task_id,
                "blocked_path": proxy.blocked_path,
                "blocked_thread": proxy.blocked_thread,
                "main_thread": main_thread,
                "terminal_status": terminal.status.value,
                "resource_writes_after_release": len(proxy.writes),
                "late_writes": len(late),
            },
        )
        lease = await agfs.pathlock_acquire_tree(prefix.rstrip("/"))
        await agfs.pathlock_release(lease)
        # The gate opens once the Task is terminal.
        assert await _wait_until(lambda: _not_active(tracker, task_id, ctx), timeout=5)
    finally:
        proxy.release.set()
        agfs._client = proxy._native


async def _not_active(tracker, task_id, ctx):
    task = await tracker.get(task_id, account_id=ctx.account_id, user_id=ctx.user.user_id)
    return task is not None and not task.is_active()
