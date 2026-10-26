"""Real-native regressions for the preset initializer's cold root phase."""
import pytest

from openviking.core.directories import DirectoryInitializer
from openviking.core.namespace import canonical_user_root, is_session_uri
from openviking.server.identity import RequestContext, Role
from openviking.storage.errors import LockAcquisitionError
from openviking_cli.session.user_id import UserIdentifier
from tests.storage.test_transfer_merge_binding import binding_fs
from tests.unit.test_directory_initializer import _FakeVikingDB


def context(account):
    return RequestContext(user=UserIdentifier(account, "actor"), role=Role.ROOT)


async def test_initializer_cold_workspace_for_thirty_accounts_and_repeat(binding_fs):
    fs = binding_fs
    failures = []
    for index in range(30):
        ctx = context(f"initialized-{index}")
        db = _FakeVikingDB()
        initializer = DirectoryInitializer(db, viking_fs=fs)
        try:
            account_count, user_count = await initializer.initialize_account_workspace(ctx)
        except Exception as exc:
            failures.append((index, type(exc).__name__, str(exc)))
            continue
        assert account_count == 6 and user_count == 7
        roots, children = initializer._account_directory_targets(ctx)
        user_root, user_children = initializer._user_directory_targets(ctx)
        expected = {target.uri for target in (*roots, *children, user_root, *user_children)}
        vectorized = {uri for uri in expected if not is_session_uri(uri)}
        assert {msg.context_data["uri"] for msg in db.embedding_messages} == vectorized
        assert len(db.embedding_messages) == 2 * len(vectorized)
        assert len(db.get_calls) == 1
        for uri in expected:
            assert (await fs.read_file(f"{uri}/.abstract.md", ctx=ctx))
            assert (await fs.read_file(f"{uri}/.overview.md", ctx=ctx))
        # Existing parents and all preset bytes remain idempotent.
        assert await initializer.initialize_account_workspace(ctx) == (0, 0)
        assert len(db.get_calls) == 1
        assert len(db.embedding_messages) == 2 * len(vectorized)
    assert not failures, f"initializer failed fresh accounts: {failures!r}"


async def test_initializer_foreign_account_tree_refuses_without_creating_children(binding_fs):
    fs, ctx = binding_fs, context("foreign-initializer")
    db = _FakeVikingDB()
    initializer = DirectoryInitializer(db, viking_fs=fs)
    account = fs._uri_to_path("viking://", ctx=ctx)
    lease = await fs._async_agfs.pathlock_acquire_tree(account)
    try:
        with pytest.raises(OSError) as raised:
            await initializer.initialize_account_workspace(ctx)
        assert isinstance(raised.value.__cause__, LockAcquisitionError)
        for uri in ("viking://resources", "viking://agent", "viking://user", canonical_user_root(ctx)):
            assert not await fs.exists(uri, ctx=ctx)
        assert not db.embedding_messages and not db.get_calls
    finally:
        await fs._async_agfs.pathlock_release(lease)


@pytest.mark.parametrize("blocked", [("viking://agent",), ("viking://resources", "viking://agent")])
async def test_initializer_preserves_and_indexes_successes_then_raises_first_native_error(binding_fs, blocked):
    fs, ctx = binding_fs, context("partial-initializer")
    db = _FakeVikingDB()
    initializer = DirectoryInitializer(db, viking_fs=fs)
    leases = []
    try:
        for uri in blocked:
            leases.append(await fs._async_agfs.pathlock_acquire_tree(fs._uri_to_path(uri, ctx=ctx)))
        with pytest.raises(OSError) as raised:
            await initializer.initialize_account_workspace(ctx)
        assert blocked[0] in str(raised.value)
        assert isinstance(raised.value.__cause__, LockAcquisitionError)
        user_root = canonical_user_root(ctx)
        successful = {"viking://resources", "viking://agent", user_root} - set(blocked)
        assert {msg.context_data["uri"] for msg in db.embedding_messages} == successful
        assert len(db.embedding_messages) == 2 * len(successful)
        assert len(db.get_calls) == 1
        for uri in successful:
            assert await fs.read_file(f"{uri}/.abstract.md", ctx=ctx)
            assert await fs.read_file(f"{uri}/.overview.md", ctx=ctx)
        for uri in blocked:
            assert await fs._async_agfs.ls(fs._uri_to_path(uri, ctx=ctx)) == []
        # A root failure must not run the child phase, even below successful roots.
        assert not await fs.exists("viking://agent/skills", ctx=ctx)
        assert not await fs.exists(user_root + "/memories", ctx=ctx)
    finally:
        for lease in reversed(leases):
            await fs._async_agfs.pathlock_release(lease)
