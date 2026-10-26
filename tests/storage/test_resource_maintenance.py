"""Resources-only maintenance with enabled ACLs, real local stores and RAGFS."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from openviking.server.api_keys.legacy import ACCOUNTS_PATH
from openviking.server.auth import get_request_context
from openviking.server.identity import RequestContext, Role
from openviking.server.routers import filesystem
from openviking.service.fs_service import FSService
from openviking.service.task_store import PersistentTaskStore
from openviking.service.task_tracker import TaskTracker
from openviking.session.memory.memory_updater import MemoryUpdater
from openviking.storage.acl import AclMode
from openviking.storage.errors import LockAcquisitionError
from openviking_cli.exceptions import OpenVikingError
from openviking_cli.session.user_id import UserIdentifier
from tests.storage.test_tree_replace_structured import (
    binding_fs as binding_fs,
)
from tests.storage.test_tree_replace_structured import (
    enabled_tree as _enabled_tree,
)
from tests.storage.test_tree_replace_structured import (
    indexed_fs as indexed_fs,
)
from tests.storage.test_tree_replace_structured import (
    raw,
)

# Export the shared fixture while keeping its import separate from test arguments.
enabled_tree = _enabled_tree


async def seed(backend, uri, ctx, key="orphan"):
    await backend.upsert(
        {
            "id": key,
            "uri": uri,
            "level": 2,
            "account_id": ctx.account_id,
            "vector": [0.1] * 4,
            "abstract": "fixture",
        },
        ctx=ctx,
    )


def app_for(service, ctx, monkeypatch):
    app = FastAPI()
    app.include_router(filesystem.router)
    app.dependency_overrides[get_request_context] = lambda: ctx
    monkeypatch.setattr(filesystem, "get_service", lambda: SimpleNamespace(fs=service))

    @app.exception_handler(OpenVikingError)
    async def error(_request, exc):
        return JSONResponse(
            status_code=409,
            content={
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        )

    return app


@pytest.mark.parametrize(
    "uri",
    [
        "viking://user/default/memories/a",
        "viking://agent/default/memories/a",
        "viking://session/s1",
        "viking://resources",
        "viking://resources/",
        "viking://resources/a/../b",
        "viking://resources/a//b",
        "viking://resources/a/",
        "viking://resources/%2e%2e/b",
        "viking://resources/a?recursive=true",
        "viking://resources/a\\b",
        "viking://resources/{user}",
        "viking://resources/a\n",
        "a" * 4096,
        "0" * 32,
        "viking://resources/a/.path.ovlock",
    ],
)
async def test_invalid_targets_before_backend(uri, monkeypatch):
    service = FSService()
    backend = AsyncMock(side_effect=AssertionError("backend reached"))
    monkeypatch.setattr(service, "_ensure_initialized", backend)
    for operation in (service.remove_resource_only, service.prune_resource_index):
        with pytest.raises(OpenVikingError):
            await operation(uri, ctx=None)
    backend.assert_not_called()


@pytest.mark.parametrize(
    "route,body",
    [
        ("remove", {"uri": "viking://resources/a", "wait": True}),
        ("remove", {"uri": "viking://resources/a", "recursive": "true"}),
        ("prune-index", {"uri": "viking://resources/a", "recursive": True}),
        ("prune-index", {"uri": "viking://resources/a", "all_accounts": True}),
    ],
)
async def test_http_strict_models_before_service(route, body, monkeypatch):
    service = SimpleNamespace(remove_resource_only=AsyncMock(), prune_resource_index=AsyncMock())
    app = app_for(service, None, monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://fixture"
    ) as client:
        response = await client.post("/api/v1/fs/resources-only/" + route, json=body)
    assert response.status_code == 422
    service.remove_resource_only.assert_not_called()
    service.prune_resource_index.assert_not_called()


async def test_public_ordinary_delete_keeps_cleanup_new_route_has_no_producers(
    enabled_tree, monkeypatch
):
    fs, backend, ctx, target, *_ = enabled_tree
    links = SimpleNamespace(before_resource_delete=AsyncMock(return_value={}))
    service = FSService(fs, resource_memory_link_service=links)
    refresh = AsyncMock()
    watch = AsyncMock()
    monkeypatch.setattr(service, "_enqueue_delete_refresh", refresh)
    monkeypatch.setattr(service, "_sync_watch_after_rm", watch)
    # The actual ordinary HTTP handler still calls tracked FSService.rm.
    tracker = TaskTracker(PersistentTaskStore(fs._async_agfs))
    monkeypatch.setattr("openviking.service.task_tracker._instance", tracker)
    app = app_for(service, ctx, monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://fixture"
    ) as client:
        result = await client.delete("/api/v1/fs", params={"uri": target + "/one.md"})
        assert result.status_code == 200, result.text
        links.before_resource_delete.assert_awaited_once()
        refresh.assert_awaited_once()
        watch.assert_awaited_once()
        links.before_resource_delete.reset_mock()
        refresh.reset_mock()
        watch.reset_mock()
        service.compile_versions = SimpleNamespace(
            before_remove=AsyncMock(side_effect=AssertionError("version effect"))
        )
        create_task = AsyncMock(wraps=tracker.create)
        monkeypatch.setattr(tracker, "create", create_task)
        memory_overview = AsyncMock(side_effect=AssertionError("memory overview effect"))
        monkeypatch.setattr(MemoryUpdater, "refresh_schema_overview", memory_overview)
        result = await client.post(
            "/api/v1/fs/resources-only/remove",
            json={
                "uri": target,
                "recursive": True,
            },
        )
        assert result.status_code == 200, result.text
        assert result.json()["result"] == {"uri": target, "outcome": "removed"}
    assert not await fs.exists(target, ctx=ctx)
    assert not await backend.get_context_by_uri(target, ctx=ctx)
    links.before_resource_delete.assert_not_called()
    refresh.assert_not_called()
    watch.assert_not_called()
    service.compile_versions.before_remove.assert_not_called()
    create_task.assert_not_called()
    memory_overview.assert_not_called()


async def test_directory_requires_explicit_recursive_before_index_mutation(
    enabled_tree, monkeypatch
):
    fs, backend, ctx, target, *_ = enabled_tree
    delete = AsyncMock(wraps=fs._delete_from_vector_store)
    monkeypatch.setattr(fs, "_delete_from_vector_store", delete)
    with pytest.raises(OpenVikingError) as error:
        await FSService(fs).remove_resource_only(target, ctx)
    assert error.value.code == "INVALID_ARGUMENT"
    assert error.value.details["outcome"] == "not_started"
    delete.assert_not_called()
    assert await backend.get_context_by_uri(target, ctx=ctx)


async def test_exact_orphan_prune_keeps_descendants_siblings_and_foreign_account(enabled_tree):
    fs, backend, ctx, target, *_ = enabled_tree
    uri = target + "/absent"
    foreign = replace(ctx, user=UserIdentifier("other-account", "owner"))
    # ROOT with a foreign account_id never permits a cross-account filter.
    await seed(backend, uri, ctx)
    await seed(backend, uri + "/child", ctx, "descendant")
    await seed(backend, uri + "-sibling", ctx, "sibling")
    await seed(backend, uri, foreign, "foreign")
    await seed(backend, uri + "/", foreign, "foreign-slash")
    result = await FSService(fs).prune_resource_index(uri, ctx)
    assert result == {"uri": uri, "outcome": "pruned"}
    assert not await fs.exists(uri, ctx=ctx)
    assert not await backend.get(["orphan"], ctx=ctx)
    assert await backend.get(["descendant", "sibling"], ctx=ctx)
    assert len(await backend.get(["foreign", "foreign-slash"], ctx=foreign)) == 2


@pytest.mark.parametrize("kind", ["file", "directory", "empty_directory", "stale_absence"])
async def test_prune_refuses_every_existing_node_under_final_coverage(
    enabled_tree, monkeypatch, kind
):
    fs, backend, ctx, target, *_ = enabled_tree
    uri = target + "/occupied"
    agfs = fs._async_agfs
    path = fs._uri_to_path(uri, ctx=ctx)
    if kind == "file":
        await fs.write_file_bytes(uri, b"keep", ctx=ctx)
    elif kind != "stale_absence":
        await fs.mkdir(uri, ctx=ctx)
        if kind == "directory":
            await fs.write_file_bytes(uri + "/child", b"keep", ctx=ctx)
    else:
        assert not await fs.exists(uri, ctx=ctx)
        acquire = agfs.pathlock_acquire_batch

        async def create_after_acquire(*args, **kwargs):
            lease = await acquire(*args, **kwargs)
            await agfs.mkdir(path, fs_ctx=fs._pathlock_fs_ctx(ctx, lease))
            return lease

        monkeypatch.setattr(agfs, "pathlock_acquire_batch", create_after_acquire)
    await seed(backend, uri, ctx)
    deletion = AsyncMock(wraps=fs._delete_from_vector_store)
    monkeypatch.setattr(fs, "_delete_from_vector_store", deletion)
    with pytest.raises(OpenVikingError) as error:
        await FSService(fs).prune_resource_index(uri, ctx)
    assert error.value.code == "CONFLICT"
    assert error.value.details["outcome"] == "not_started"
    deletion.assert_not_called()
    assert await fs.exists(uri, ctx=ctx)
    assert await backend.get(["orphan"], ctx=ctx)


@pytest.mark.parametrize("operation", ["remove", "prune"])
async def test_descendant_and_exact_orphan_acl_denials(enabled_tree, operation):
    fs, backend, ctx, target, *_ = enabled_tree
    user = RequestContext(user=UserIdentifier(ctx.account_id, "maintainer"), role=Role.USER)
    owner = {"principal": "user:" + ctx.user.user_id, "level": "manage"}
    maintainer = {"principal": "user:maintainer", "level": "manage"}
    await fs.set_acl(target, [owner, maintainer], ctx=ctx, acl_mode=AclMode.RESTRICTED)
    uri = target + "/branch" if operation == "remove" else target + "/orphan"
    if operation == "prune":
        await seed(backend, uri, ctx)
        # Model a persisted orphan policy through the real ACL index manager;
        # the filesystem remains absent throughout, with no placeholder node.
        await fs.acl_manager.set_acl(uri, [owner], ctx, acl_mode=AclMode.RESTRICTED)
    else:
        await fs.set_acl(uri, [owner], ctx=ctx, acl_mode=AclMode.RESTRICTED)
    with pytest.raises(OpenVikingError) as error:
        if operation == "remove":
            await FSService(fs).remove_resource_only(target, user, recursive=True)
        else:
            await FSService(fs).prune_resource_index(uri, user)
    assert error.value.code == "PERMISSION_DENIED"
    assert await raw(fs, target + "/one.md", ctx) == b"old one"
    assert await backend.get_context_by_uri(uri, ctx=ctx)


@pytest.mark.parametrize("operation", ["remove", "prune"])
@pytest.mark.parametrize("change", ["deleting", "recreate", "no_manager"])
@pytest.mark.parametrize("boundary", ["outer", "inner"])
async def test_root_account_lifetime_never_bypassed(
    enabled_tree, monkeypatch, operation, change, boundary
):
    fs, backend, ctx, target, *_ = enabled_tree
    uri = target if operation == "remove" else target + "/orphan"
    if operation == "prune":
        await seed(backend, uri, ctx)
    manager, agfs = fs._account_identity_manager, fs._async_agfs
    acquire = agfs.pathlock_acquire_batch
    if change == "no_manager":
        fs.set_account_identity_manager(None)
    else:
        calls = 0

        async def change_account(*args, **kwargs):
            nonlocal calls
            lease = await acquire(*args, **kwargs)
            calls += 1
            if calls != (1 if boundary == "outer" else 2):
                return lease
            if change == "deleting":
                await manager.begin_deletion(
                    ctx.account_id,
                    None,
                    task_id="fixture-delete",
                    owner_account_id=ctx.account_id,
                    owner_user_id=ctx.user.user_id,
                )
            else:
                data = json.loads(await agfs.cat(ACCOUNTS_PATH))
                data["accounts"][ctx.account_id]["creation_id"] = str(uuid4())
                await agfs.write(ACCOUNTS_PATH, json.dumps(data).encode())
            return lease

        monkeypatch.setattr(agfs, "pathlock_acquire_batch", change_account)
    deletion = AsyncMock(wraps=fs._delete_from_vector_store)
    monkeypatch.setattr(fs, "_delete_from_vector_store", deletion)
    with pytest.raises(OpenVikingError):
        if operation == "remove":
            await FSService(fs).remove_resource_only(uri, ctx, recursive=True)
        else:
            await FSService(fs).prune_resource_index(uri, ctx)
    deletion.assert_not_called()
    assert await raw(fs, target + "/one.md", ctx) == b"old one"


@pytest.mark.parametrize("operation", ["remove", "prune"])
async def test_foreign_root_lease_refuses_without_owner_disclosure(enabled_tree, operation):
    fs, backend, ctx, target, *_ = enabled_tree
    uri = target if operation == "remove" else target + "/orphan"
    await seed(backend, uri, ctx)
    root = fs._uri_to_path("viking://resources", ctx=ctx)
    lease = await fs._async_agfs.pathlock_acquire_tree(root)
    try:
        with pytest.raises(OpenVikingError) as error:
            if operation == "remove":
                await FSService(fs).remove_resource_only(uri, ctx, recursive=True)
            else:
                await FSService(fs).prune_resource_index(uri, ctx)
        assert error.value.details == {"uri": uri, "outcome": "not_started"}
        assert await backend.get(["orphan"], ctx=ctx)
    finally:
        await fs._async_agfs.pathlock_release(lease)


@pytest.mark.parametrize("failure", ["index", "filesystem", "ownership", "cancel"])
async def test_partial_failure_and_cancellation_release_coverage_truthfully(
    enabled_tree, monkeypatch, failure
):
    fs, backend, ctx, target, *_ = enabled_tree
    agfs = fs._async_agfs
    uri = target + "/one.md"
    await seed(backend, uri, ctx)
    if failure == "index":

        async def fail_index(*args, **kwargs):
            raise OSError("foreign secret index metadata")

        monkeypatch.setattr(fs, "_delete_from_vector_store", fail_index)
    else:
        original = agfs.rm

        async def fail(path, *args, **kwargs):
            if path == fs._uri_to_path(uri, ctx=ctx):
                if failure == "cancel":
                    raise asyncio.CancelledError()
                if failure == "ownership":
                    raise LockAcquisitionError("foreign lease owner secret")
                raise OSError("foreign secret filesystem metadata")
            return await original(path, *args, **kwargs)

        monkeypatch.setattr(agfs, "rm", fail)
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await FSService(fs).remove_resource_only(uri, ctx)
    else:
        with pytest.raises(OpenVikingError) as error:
            await FSService(fs).remove_resource_only(uri, ctx)
        assert error.value.details == {"uri": uri, "outcome": "unknown"}
        assert "secret" not in str(error.value)
    assert await raw(fs, uri, ctx) == b"old one"
    assert bool(await backend.get(["orphan"], ctx=ctx)) == (failure == "index")
    # Both inner owner extension and outer coverage must have been released.
    lease = await agfs.pathlock_acquire_tree(fs._uri_to_path("viking://resources", ctx=ctx))
    await agfs.pathlock_release(lease)


async def test_manage_authorized_user_can_prune_exact_orphan_over_http(enabled_tree, monkeypatch):
    fs, backend, ctx, target, *_ = enabled_tree
    user = RequestContext(user=UserIdentifier(ctx.account_id, "maintainer"), role=Role.USER)
    await fs.set_acl(
        target,
        [
            {"principal": "user:" + ctx.user.user_id, "level": "manage"},
            {"principal": "user:maintainer", "level": "manage"},
        ],
        ctx=ctx,
        acl_mode=AclMode.RESTRICTED,
    )
    uri = target + "/orphan"
    await seed(backend, uri, ctx)
    app = app_for(FSService(fs), user, monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://fixture"
    ) as client:
        response = await client.post("/api/v1/fs/resources-only/prune-index", json={"uri": uri})
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {"uri": uri, "outcome": "pruned"}
    assert not await backend.get(["orphan"], ctx=ctx)
    assert not await fs.exists(uri, ctx=ctx)


async def test_remove_absent_node_does_not_prune_index(enabled_tree):
    fs, backend, ctx, target, *_ = enabled_tree
    uri = target + "/orphan"
    await seed(backend, uri, ctx)
    with pytest.raises(OpenVikingError) as error:
        await FSService(fs).remove_resource_only(uri, ctx, recursive=True)
    assert error.value.code == "NOT_FOUND"
    assert error.value.details["outcome"] == "not_started"
    assert await backend.get(["orphan"], ctx=ctx)


async def test_actual_owner_coverage_loss_after_index_delete_has_no_success(
    enabled_tree, monkeypatch
):
    fs, backend, ctx, target, *_ = enabled_tree
    uri = target + "/one.md"
    await seed(backend, uri, ctx)
    agfs, leases = fs._async_agfs, []
    acquire, release = agfs.pathlock_acquire_batch, agfs.pathlock_release
    delete = fs._delete_from_vector_store

    async def remember(*args, **kwargs):
        lease = await acquire(*args, **kwargs)
        leases.append(lease)
        return lease

    async def lose_after_index(*args, **kwargs):
        await delete(*args, **kwargs)
        for lease in reversed(leases):
            await release(lease)

    monkeypatch.setattr(agfs, "pathlock_acquire_batch", remember)
    monkeypatch.setattr(fs, "_delete_from_vector_store", lose_after_index)
    with pytest.raises(OpenVikingError) as error:
        await FSService(fs).remove_resource_only(uri, ctx)
    assert error.value.details == {"uri": uri, "outcome": "unknown"}
    assert await raw(fs, uri, ctx) == b"old one"
    assert not await backend.get(["orphan"], ctx=ctx)
    lease = await acquire(
        [{"path": fs._uri_to_path("viking://resources", ctx=ctx), "kind": "tree"}]
    )
    await release(lease)


@pytest.mark.parametrize("root", [False, True])
async def test_prune_native_uri_equivalence_keeps_same_acl_identity(enabled_tree, root):
    fs, backend, ctx, target, *_ = enabled_tree
    user = ctx if root else RequestContext(
        user=UserIdentifier(ctx.account_id, "maintainer"), role=Role.USER,
    )
    owner = {"principal": "user:" + ctx.user.user_id, "level": "manage"}
    await fs.set_acl(
        target, [owner, {"principal": "user:maintainer", "level": "manage"}],
        ctx=ctx, acl_mode=AclMode.RESTRICTED,
    )
    uri = target + "/orphan"
    await seed(backend, uri, ctx, "canonical")
    await seed(backend, uri + "/", ctx, "slash")
    result = await FSService(fs).prune_resource_index(uri, user)
    assert result == {"uri": uri, "outcome": "pruned"}
    assert not await backend.get(["canonical", "slash"], ctx=ctx)
    assert not await fs.exists(uri, ctx=ctx)


@pytest.mark.parametrize("operation", ["remove", "prune"])
async def test_caller_cancellation_after_index_delete_releases_both_leases(
    enabled_tree, monkeypatch, operation
):
    fs, backend, ctx, target, *_ = enabled_tree
    uri = target + ("/one.md" if operation == "remove" else "/orphan")
    await seed(backend, uri, ctx)
    reached, wait = asyncio.Event(), asyncio.Event()
    delete = fs._delete_from_vector_store

    async def pause_after_index(*args, **kwargs):
        await delete(*args, **kwargs)
        reached.set()
        await wait.wait()

    monkeypatch.setattr(fs, "_delete_from_vector_store", pause_after_index)
    service = FSService(fs)
    call = (
        service.remove_resource_only(uri, ctx)
        if operation == "remove"
        else service.prune_resource_index(uri, ctx)
    )
    task = asyncio.create_task(call)
    try:
        await asyncio.wait_for(reached.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert not await backend.get(["orphan"], ctx=ctx)
    if operation == "remove":
        assert await raw(fs, uri, ctx) == b"old one"
    else:
        assert not await fs.exists(uri, ctx=ctx)
    lease = await fs._async_agfs.pathlock_acquire_tree(
        fs._uri_to_path("viking://resources", ctx=ctx)
    )
    await fs._async_agfs.pathlock_release(lease)
