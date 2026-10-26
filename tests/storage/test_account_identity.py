# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Internal account lifetimes, explicit backfill and authoritative rechecks."""

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from openviking.server.account_stores.base import AccountStore
from openviking.server.account_stores.models import AccountIdentity
from openviking.server.account_stores.mysql import MySQLAccountStore
from openviking.server.account_stores.redis_mysql import RedisMySQLAccountStore
from openviking.server.api_keys.legacy import ACCOUNTS_PATH, FileStore
from openviking.server.api_keys.new import APIKeyManager
from openviking.server.identity import RequestContext, Role
from openviking.service.account_identity import capture_account_identity, recheck_account_identity
from openviking_cli.exceptions import ConflictError, FailedPreconditionError
from openviking_cli.session.user_id import UserIdentifier
from tests.server.account_store_fakes import InMemoryAGFS


class ObservedAGFS(InMemoryAGFS):
    def __init__(self):
        super().__init__()
        self.reads = []
        self.writes = []
        self.cached_registry = None

    def read(self, path, ctx=None):
        self.reads.append((path, dict(ctx or {})))
        if path == ACCOUNTS_PATH and self.cached_registry is not None:
            if (ctx or {}).get("bypass_cache") != "true":
                return self.cached_registry
        return super().read(path, ctx)

    def write(self, path, content, ctx=None):
        self.writes.append((path, dict(ctx or {})))
        return super().write(path, content, ctx)


def _seed(agfs, data):
    agfs.write(ACCOUNTS_PATH, json.dumps(data).encode())
    agfs.writes.clear()


def _store(agfs, **kwargs):
    return FileStore(SimpleNamespace(agfs=agfs), **kwargs)


async def test_legacy_missing_identity_never_backfills_on_load_or_reads():
    agfs = ObservedAGFS()
    data = {"extra": {"preserve": True}, "accounts": {"old": {
        "created_at": "old", "unknown": [1, 2],
        "deletion": {"task_id": "pending", "opaque": "keep"},
    }}}
    _seed(agfs, data)
    store = _store(agfs)
    await store.load()
    try:
        before = agfs._files[ACCOUNTS_PATH]
        await store._reload()
        with pytest.raises(FailedPreconditionError, match="explicit initialization"):
            await store.get_account_identity("old")
        assert agfs._files[ACCOUNTS_PATH] == before
        assert not agfs.writes
        assert "creation_id" not in await store.get_account("old")
        assert "creation_identity" not in await store.get_account("old")
    finally:
        await store.close()


async def test_explicit_backfill_is_locked_preserves_records_and_is_idempotent():
    agfs = ObservedAGFS()
    existing = str(uuid4())
    before = {"top": {"version": 19}, "accounts": {
        "old": {"created_at": "same", "unknown": {"nested": [1, 2]}},
        "fenced": {"created_at": "same", "deletion": {
            "task_id": "pending", "owner_account_id": "system", "unknown": "keep"
        }},
        "current": {"created_at": "same", "creation_id": existing, "extra": 9},
    }}
    _seed(agfs, before)
    first = _store(agfs, params={"initialize_creation_ids": True})
    second = _store(agfs, params={"initialize_creation_ids": True})
    await asyncio.gather(first.load(), second.load())
    try:
        after = json.loads(agfs._files[ACCOUNTS_PATH])
        assert len(agfs.writes) == 1
        assert agfs.writes[0][1]["lease_ref"]
        assert after["top"] == before["top"]
        assert set(after["accounts"]) == set(before["accounts"])
        for account, original in before["accounts"].items():
            record = after["accounts"][account]
            assert UUID(record["creation_id"]).version == 4
            assert {k: v for k, v in record.items() if k != "creation_id"} == {
                k: v for k, v in original.items() if k != "creation_id"
            }
            assert await first.get_account_identity(account) == await second.get_account_identity(account)
        assert after["accounts"]["current"]["creation_id"] == existing
        assert (await first.get_account_identity("fenced")).deleting
        await first._reload()
        restarted = _store(agfs)
        await restarted.load()
        await restarted.close()
        assert json.loads(agfs._files[ACCOUNTS_PATH]) == after
        assert len(agfs.writes) == 1
    finally:
        await first.close()
        await second.close()


@pytest.mark.parametrize("invalid", ["", "timestamp", None, 42])
async def test_backfill_refuses_invalid_existing_identity_without_partial_write(invalid):
    agfs = ObservedAGFS()
    data = {"accounts": {"missing": {"created_at": "x"}, "invalid": {
        "created_at": "x", "creation_id": invalid
    }}}
    _seed(agfs, data)
    with pytest.raises(FailedPreconditionError):
        await _store(agfs, params={"initialize_creation_ids": True}).load()
    assert json.loads(agfs._files[ACCOUNTS_PATH]) == data
    assert not agfs.writes


async def test_new_creation_and_recreation_use_fresh_uuid_with_identical_timestamps(monkeypatch):
    from openviking.server.api_keys import legacy

    class FixedClock:
        @staticmethod
        def now(tz):
            return datetime(2026, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(legacy, "datetime", FixedClock)
    agfs = ObservedAGFS()
    store = _store(agfs)
    await store.load()
    try:
        await store.create_account("account", "admin")
        old = await store.get_account_identity("account")
        timestamp = (await store.get_account("account"))["created_at"]
        await store.delete_account("account")
        await store.create_account("account", "admin")
        new = await store.get_account_identity("account")
        assert new.creation_identity != old.creation_identity
        assert (await store.get_account("account"))["created_at"] == timestamp
        await store.ensure_trusted_identities({"trusted": {"user"}})
        for account in ("default", "account", "trusted"):
            identity = await store.get_account_identity(account)
            assert UUID(identity.creation_identity.removeprefix("legacy:")).version == 4
        with pytest.raises(FrozenInstanceError):
            new.creation_identity = "changed"
    finally:
        await store.close()


async def test_uuid_preserved_across_registry_updates_unknown_fields_and_fences():
    agfs = ObservedAGFS()
    store = _store(agfs)
    await store.load()
    try:
        await store.create_account("account", "admin")
        admitted = await store.get_account_identity("account")
        data = json.loads(agfs._files[ACCOUNTS_PATH])
        data["accounts"]["account"]["future_field"] = {"keep": 1}
        agfs.write(ACCOUNTS_PATH, json.dumps(data).encode())
        await store.begin_deletion(
            "account", None, task_id="task", owner_account_id="system", owner_user_id="system"
        )
        await store.replace_deletion_task(
            "account", None, expected_task_id="task", task_id="retry",
            owner_account_id="system", owner_user_id="system"
        )
        identity = await store.get_account_identity("account")
        assert identity.creation_identity == admitted.creation_identity
        assert identity.deleting
        record = json.loads(agfs._files[ACCOUNTS_PATH])["accounts"]["account"]
        assert record["future_field"] == {"keep": 1}
        assert record["deletion"]["task_id"] == "retry"
        await store._reload()
        assert await store.get_account_identity("account") == identity
    finally:
        await store.close()


async def test_identity_read_bypasses_watcher_and_backend_cache_and_missing_registry():
    agfs = ObservedAGFS()
    store = _store(agfs, watch_enabled=True)
    await store.load()
    try:
        old = await store.get_account_identity("default")
        agfs.cached_registry = agfs._files[ACCOUNTS_PATH]
        data = json.loads(agfs.cached_registry)
        data["accounts"]["default"]["creation_id"] = str(uuid4())
        agfs.write(ACCOUNTS_PATH, json.dumps(data).encode())
        assert store._watcher_running()
        assert (await store.get_account_identity("default")).creation_identity != old.creation_identity
        assert agfs.reads[-1] == (ACCOUNTS_PATH, {
            "account_id": "_system", "bypass_cache": "true"
        })
        del agfs._files[ACCOUNTS_PATH]
        assert await store.get_account_identity("default") is None
        assert not store._accounts and not store._prefix_index
        assert not store._user_group_ids
    finally:
        await store.close()


@pytest.mark.parametrize("role", [Role.ROOT, Role.USER])
@pytest.mark.parametrize("state", ["absent", "deleting", "recreated"])
async def test_admitted_operation_refuses_final_lease_after_account_change(role, state, tmp_path):
    import ragfs_python

    from openviking.pyagfs import AsyncAGFSClient

    client = ragfs_python.RAGFSBindingClient()
    client.mount("localfs", "/local", {"local_dir": str(tmp_path)})
    agfs = AsyncAGFSClient(client)
    store = _store(client)
    manager = APIKeyManager("synthetic-root", store)
    await manager.load()
    ctx = RequestContext(user=UserIdentifier("account", "user"), role=role)
    try:
        await store.create_account("account", "admin")
        admitted = await capture_account_identity(manager, ctx.account_id)
        if state == "deleting":
            await store.begin_deletion(
                "account", None, task_id="task", owner_account_id="system", owner_user_id="system"
            )
        else:
            await store.delete_account("account")
            await agfs.rm("/local/account", recursive=True)
            if state == "recreated":
                await store.create_account("account", "admin")
        # Final coverage may create this empty namespace after teardown.
        lease = await agfs.pathlock_acquire_tree("/local/account/resources")
        foreign = tmp_path / "account/resources/foreign-empty"
        foreign.mkdir()
        writes = AsyncMock()
        try:
            with pytest.raises(ConflictError):
                await recheck_account_identity(manager, ctx.account_id, admitted)
                await writes()
        finally:
            await agfs.pathlock_release(lease)
        writes.assert_not_awaited()
        assert foreign.is_dir()
        assert not any(p.is_file() for p in foreign.rglob("*"))
    finally:
        await manager.close()


async def test_unsupported_and_unknown_reads_fail_closed():
    with pytest.raises(FailedPreconditionError):
        await AccountStore.get_account_identity(object(), "account")
    manager = SimpleNamespace(get_account_identity=AsyncMock(side_effect=OSError("unavailable")))
    with pytest.raises(OSError, match="unavailable"):
        await capture_account_identity(manager, "account")
    manager.get_account_identity = AsyncMock(return_value=AccountIdentity("", False))
    with pytest.raises(ConflictError):
        await capture_account_identity(manager, "account")


async def test_mysql_identity_uses_new_scoped_session_and_redis_bypasses_cache():
    snapshots = [
        SimpleNamespace(account_pk=12, deletion_task_id=None),
        SimpleNamespace(account_pk=12, deletion_task_id="delete"),
        None,
        SimpleNamespace(account_pk=13, deletion_task_id=None),
    ]
    sessions = []

    @asynccontextmanager
    async def factory():
        row = snapshots[len(sessions)]
        async def scalar(statement):
            params = statement.compile().params
            assert "tenant" in params.values() and "account" in params.values()
            return row
        session = SimpleNamespace(scalar=scalar)
        sessions.append(session)
        yield session

    mysql = object.__new__(MySQLAccountStore)
    mysql._sessions = factory
    mysql._mysql_runtime = SimpleNamespace(run=lambda operation: operation())
    mysql._service_name = "tenant"
    redis = object.__new__(RedisMySQLAccountStore)
    redis._mysql = mysql
    redis._read_through = AsyncMock(side_effect=AssertionError("cache must not run"))
    manager = APIKeyManager("synthetic-root", redis)
    first = await capture_account_identity(manager, "account")
    assert first == AccountIdentity("mysql:tenant:12", False)
    assert (await manager.get_account_identity("account")).deleting
    assert await manager.get_account_identity("account") is None
    with pytest.raises(ConflictError):
        await recheck_account_identity(manager, "account", first)
    assert len(sessions) == 4 and len({id(s) for s in sessions}) == 4
    redis._read_through.assert_not_awaited()

async def test_stale_registry_writer_cannot_resurrect_or_delete_another_lifetime():
    agfs = ObservedAGFS()
    writer = _store(agfs)
    stale = _store(agfs, watch_enabled=True)
    await writer.load()
    await writer.create_account("account", "admin")
    await stale.load()
    try:
        old = await stale.get_account_identity("account")
        await writer.delete_account("account")
        with pytest.raises(FailedPreconditionError, match="refusing to recreate"):
            await stale.begin_deletion(
                "account", None, task_id="stale", owner_account_id="system", owner_user_id="system"
            )
        assert await writer.get_account_identity("account") is None
        await writer.create_account("account", "admin")
        new = await writer.get_account_identity("account")
        assert new.creation_identity != old.creation_identity
        with pytest.raises(FailedPreconditionError, match="refusing stale deletion"):
            await stale.delete_account("account")
        assert await writer.get_account_identity("account") == new
    finally:
        await writer.close()
        await stale.close()


async def test_stale_prebackfill_writer_refuses_to_erase_assigned_uuid():
    agfs = ObservedAGFS()
    _seed(agfs, {"accounts": {"old": {"created_at": "same", "unknown": 9}}})
    stale = _store(agfs, watch_enabled=True)
    await stale.load()
    migrated = _store(agfs, params={"initialize_creation_ids": True})
    await migrated.load()
    try:
        assigned = await migrated.get_account_identity("old")
        with pytest.raises(FailedPreconditionError, match="identity changed"):
            await stale.begin_deletion(
                "old", None, task_id="stale", owner_account_id="system", owner_user_id="system"
            )
        assert await migrated.get_account_identity("old") == assigned
        assert json.loads(agfs._files[ACCOUNTS_PATH])["accounts"]["old"]["unknown"] == 9
    finally:
        await stale.close()
        await migrated.close()


@pytest.mark.parametrize("raw", [
    b"null", b"[]", b'{"accounts":null}', b'{"accounts":{"old":null}}',
])
async def test_unknown_registry_shape_is_never_initialized_or_overwritten(raw):
    agfs = ObservedAGFS()
    agfs.write(ACCOUNTS_PATH, raw)
    agfs.writes.clear()
    store = _store(agfs, params={"initialize_creation_ids": True})
    with pytest.raises(FailedPreconditionError):
        await store.load()
    assert agfs._files[ACCOUNTS_PATH] == raw
    assert not agfs.writes
