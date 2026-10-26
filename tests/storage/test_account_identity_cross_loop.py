# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Real native FileStore shared by a request loop and a queue-like thread."""

import asyncio
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest

from openviking.server.api_keys.legacy import ACCOUNTS_PATH, FileStore
from openviking.server.api_keys.new import APIKeyManager
from openviking.server.identity import RequestContext, Role
from openviking.service.account_identity import capture_account_identity, recheck_account_identity
from openviking_cli.exceptions import ConflictError, FailedPreconditionError
from openviking_cli.session.user_id import UserIdentifier


class QueueLoop:
    def __enter__(self):
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()

        def run():
            asyncio.set_event_loop(self.loop)
            self.loop.call_soon(self.ready.set)
            self.loop.run_forever()
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.run_until_complete(self.loop.shutdown_default_executor())
            self.loop.close()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        assert self.ready.wait(5)
        return self

    def submit(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def __exit__(self, *args):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        assert not self.thread.is_alive()


def native_store(tmp_path, **kwargs):
    import ragfs_python

    client = ragfs_python.RAGFSBindingClient()
    client.mount("localfs", "/local", {"local_dir": str(tmp_path)})
    return FileStore(SimpleNamespace(agfs=client), **kwargs)


async def result(future):
    return await asyncio.wait_for(asyncio.wrap_future(future), 5)


async def test_actual_identity_contention_avoids_foreign_asyncio_mutex(tmp_path, monkeypatch):
    store = native_store(tmp_path)
    await store.load()
    loop = asyncio.get_running_loop()
    reached = [asyncio.Event(), asyncio.Event()]
    original = store._mutation_lock._get_loop
    calls = 0

    def observed_get_loop():
        nonlocal calls
        index = calls
        calls += 1
        try:
            return original()
        finally:
            if index < len(reached):
                loop.call_soon_threadsafe(reached[index].set)

    monkeypatch.setattr(store._mutation_lock, "_get_loop", observed_get_loop)
    original_read = store._read_json

    async def observed_read(*args, **kwargs):
        if asyncio.get_running_loop() is not loop:
            loop.call_soon_threadsafe(reached[1].set)
        return await original_read(*args, **kwargs)

    monkeypatch.setattr(store, "_read_json", observed_read)
    await store._mutation_lock.acquire()
    local = asyncio.create_task(store.get_account_identity("default"))
    try:
        await asyncio.wait_for(reached[0].wait(), 5)
        assert store._mutation_lock._loop is loop
        with QueueLoop() as queue:
            remote = queue.submit(store.get_account_identity("default"))
            try:
                # Baseline signals after its foreign mutex loop check; the fix
                # signals on its native snapshot read. Both are deterministic.
                await asyncio.wait_for(reached[1].wait(), 5)
            finally:
                store._mutation_lock.release()
            assert await result(remote) == await local
    finally:
        if store._mutation_lock.locked():
            store._mutation_lock.release()
        await asyncio.gather(local, return_exceptions=True)
        await store.close()


@pytest.mark.parametrize("role", [Role.ROOT, Role.USER])
async def test_queue_fresh_rechecks_during_mutation_reload_and_foreign_lease(tmp_path, role):
    store = native_store(tmp_path, watch_enabled=True)
    manager = APIKeyManager("synthetic-root", store)
    await manager.load()
    try:
        await store.create_account("account", "admin")
        ctx = RequestContext(user=UserIdentifier("account", "user"), role=role)
        admitted = await capture_account_identity(manager, ctx.account_id)
        with QueueLoop() as queue:

            async def check():
                # Both roles use the same final-check helper in production.
                await recheck_account_identity(manager, ctx.account_id, admitted)

            await result(queue.submit(check()))
            # An actual foreign registry lease blocks a create holding the
            # mutation mutex. Queue reads must wait without blocking either loop.
            lease = await store._async_agfs.pathlock_acquire_exact(ACCOUNTS_PATH)
            acquiring = asyncio.Event()
            original_acquire = store._async_agfs.pathlock_acquire_exact

            async def observe_acquire(*args, **kwargs):
                acquiring.set()
                return await original_acquire(*args, **kwargs)

            store._async_agfs.pathlock_acquire_exact = observe_acquire
            mutation = asyncio.create_task(store.create_account("other", "admin"))
            await asyncio.wait_for(acquiring.wait(), 5)
            remote = queue.submit(check())
            assert not mutation.done()
            assert not remote.done()
            await store._async_agfs.pathlock_release(lease)
            await asyncio.wait_for(mutation, 5)
            await result(remote)
            store._async_agfs.pathlock_acquire_exact = original_acquire
            await asyncio.gather(
                store._reload(),
                *[result(queue.submit(store.get_account_identity("account"))) for _ in range(4)],
            )
            await store.begin_deletion(
                "account",
                None,
                task_id="delete",
                owner_account_id="system",
                owner_user_id="system",
            )
            with pytest.raises(ConflictError):
                await result(queue.submit(check()))
            assert (await result(queue.submit(store.get_account_identity("account")))).deleting
            await store.finish_deletion("account", None, "delete")
            assert await result(queue.submit(store.get_account_identity("account"))) is None
            await store.create_account("account", "admin")
            recreated = await result(queue.submit(store.get_account_identity("account")))
            assert recreated.creation_identity != admitted.creation_identity
            with pytest.raises(ConflictError):
                await result(queue.submit(check()))
            assert await result(queue.submit(store.get_account_identity("unknown"))) is None

            data = await store._read_json(ACCOUNTS_PATH, bypass_cache=True)
            assert data is not None
            del data["accounts"]["account"]["creation_id"]
            await store._write_json(ACCOUNTS_PATH, data)
            before = (tmp_path / "_system/accounts.json").read_bytes()
            with pytest.raises(FailedPreconditionError, match="explicit initialization"):
                await result(queue.submit(store.get_account_identity("account")))
            assert (tmp_path / "_system/accounts.json").read_bytes() == before

            # Bypass the watcher and backend cache for out-of-band replacement.
            data["accounts"]["account"]["creation_id"] = str(uuid4())
            await store._write_json(ACCOUNTS_PATH, data)
            assert (
                await result(queue.submit(store.get_account_identity("account")))
            ).creation_identity == ("legacy:" + data["accounts"]["account"]["creation_id"])
            await store._async_agfs.rm(ACCOUNTS_PATH)
            assert await result(queue.submit(store.get_account_identity("account"))) is None
            # Foreign snapshots never mutate caches. The management-loop
            # identity read retains the original missing-registry invalidation.
            assert await store.get_account_identity("account") is None
            assert not store._accounts and not store._prefix_index and not store._user_group_ids
            await store.close()
            assert await result(queue.submit(store.get_account_identity("account"))) is None
    finally:
        await manager.close()
