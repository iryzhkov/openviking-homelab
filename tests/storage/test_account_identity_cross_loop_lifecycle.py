# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Native identity snapshots do not depend on a live request-loop dispatcher."""

import asyncio
import json
import threading

import pytest

from openviking.server.api_keys.legacy import ACCOUNTS_PATH
from tests.storage.test_account_identity_cross_loop import QueueLoop, native_store, result


def test_queue_snapshot_does_not_need_load_loop_to_run(tmp_path):
    store = native_store(tmp_path)
    asyncio.run(store.load())
    assert store._identity_loop.is_closed()
    with QueueLoop() as queue:
        identity = queue.submit(store.get_account_identity("default")).result(5)
        assert identity is not None
    asyncio.run(store.close())


async def test_queue_snapshot_waits_for_explicit_backfill_without_cache_writes(tmp_path):
    store = native_store(tmp_path)
    await store._async_agfs.write(
        ACCOUNTS_PATH, json.dumps({"accounts": {"old": {"created_at": "same"}}}).encode()
    )
    await store.load()
    migrator = native_store(tmp_path, params={"initialize_creation_ids": True})
    writing = asyncio.Event()
    proceed = asyncio.Event()
    acquiring = asyncio.Event()
    owner_loop = asyncio.get_running_loop()
    original_write = migrator._write_json
    original_acquire = store._async_agfs.pathlock_acquire_exact

    async def blocked_write(*args, **kwargs):
        writing.set()
        await proceed.wait()
        return await original_write(*args, **kwargs)

    async def observed_acquire(*args, **kwargs):
        owner_loop.call_soon_threadsafe(acquiring.set)
        return await original_acquire(*args, **kwargs)

    migrator._write_json = blocked_write
    store._async_agfs.pathlock_acquire_exact = observed_acquire
    migration = asyncio.create_task(migrator.load())
    try:
        await asyncio.wait_for(writing.wait(), 5)
        cached_accounts = store._accounts
        with QueueLoop() as queue:
            remote = queue.submit(store.get_account_identity("old"))
            try:
                await asyncio.wait_for(acquiring.wait(), 5)
                assert not remote.done()
            finally:
                proceed.set()
            await asyncio.wait_for(migration, 5)
            identity = await result(remote)
            assert identity == await migrator.get_account_identity("old")
            assert store._accounts is cached_accounts
            assert store._accounts["old"].creation_id is None
    finally:
        proceed.set()
        await asyncio.gather(migration, return_exceptions=True)
        await migrator.close()
        await store.close()


async def test_cancelled_queue_snapshot_releases_native_lease(tmp_path):
    store = native_store(tmp_path)
    await store.load()
    loop = asyncio.get_running_loop()
    reading = asyncio.Event()
    proceed = threading.Event()
    original_read = store._read_json

    async def blocked_read(*args, **kwargs):
        loop.call_soon_threadsafe(reading.set)
        await asyncio.to_thread(proceed.wait, 5)
        return await original_read(*args, **kwargs)

    store._read_json = blocked_read
    try:
        with QueueLoop() as queue:
            remote = queue.submit(store.get_account_identity("default"))
            await asyncio.wait_for(reading.wait(), 5)
            remote.cancel()
            proceed.set()
            with pytest.raises(asyncio.CancelledError):
                await result(remote)
            store._read_json = original_read
            # A real writer requires the same Exact registry lease. Its success
            # also proves cancellation did not leave the snapshot lease live.
            await asyncio.wait_for(store.create_account("aftercancel", "admin"), 5)
            await store.close()
    finally:
        proceed.set()
        await store.close()
