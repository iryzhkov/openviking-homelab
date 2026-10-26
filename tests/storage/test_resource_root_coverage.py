"""D3 resource coverage with a real account registry, filesystem and vector store.

Acquisition hooks model intervening producers; these are Python integration
checks, not qualification of the uncompiled native mkdir participation repair.
"""
import hashlib
import json
from uuid import uuid4

import pytest

from openviking.server.api_keys.legacy import ACCOUNTS_PATH
from openviking.server.identity import Role
from openviking.service import tree_replace
from openviking.storage.content_write import ContentWriteCoordinator
from openviking.storage.errors import LockAcquisitionError, ResourceBusyError
from openviking.storage.viking_fs._ops import _OpsMixin
from openviking_cli.exceptions import ConflictError, FailedPreconditionError, NotFoundError, OpenVikingError
from tests.storage.test_tree_replace_structured import (
    binding_fs, indexed_fs, enabled_tree, revision, raw, replace_case,
)


def test_requests_collapse_to_resources_tree():
    root = "/local/account/resources"
    requests = [
        {"path": root + "/a", "kind": "exact"},
        {"path": root + "/b/c", "kind": "tree"},
        {"path": root, "kind": "exact"},
        {"path": root, "kind": "tree"},
        {"path": "/local/account/other/a", "kind": "exact"},
    ]
    assert _OpsMixin._collapse_lock_requests(requests) == [
        {"path": root, "kind": "tree"},
        {"path": "/local/account/other/a", "kind": "exact"},
    ]


async def test_own_covering_lease_extension(enabled_tree):
    fs, _, ctx, target, *_ = enabled_tree
    admitted = await fs._capture_resource_identity(ctx)
    root = fs._uri_to_path("viking://resources", ctx=ctx)
    outer = await fs._async_agfs.pathlock_acquire_tree(root)
    try:
        inner = await fs._acquire_resource_coverage(
            ctx, admitted, owner_lease_ref=outer,
            requests=[{"path": fs._uri_to_path(target, ctx=ctx), "kind": "tree"}],
        )
        await fs._async_agfs.pathlock_release(inner)
        # Extension release must not invalidate its owner's coverage.
        await fs._async_agfs.mkdir(root + "/owned", fs_ctx=fs._pathlock_fs_ctx(ctx, outer))
    finally:
        await fs._async_agfs.pathlock_release(outer)


@pytest.mark.parametrize("which", ["target", "backup"])
@pytest.mark.parametrize("when", ["before", "after_acquisition", "exclusive_create"])
async def test_foreign_empty_directory_never_absent_or_cleaned(enabled_tree, monkeypatch, which, when):
    fs, _, ctx, target, staging, backup, _, _ = enabled_tree
    # Use an actually absent target, not an empty directory inferred absent.
    target = "viking://resources/published/new"
    occupied = target if which == "target" else backup
    occupied_path = fs._uri_to_path(occupied, ctx=ctx)
    staging_revision = await revision(fs, staging, ctx)
    agfs = fs._async_agfs
    acquire, mkdir = agfs.pathlock_acquire_batch, agfs.mkdir
    fired = False

    async def create(lease=None):
        nonlocal fired
        if not fired:
            fired = True
            await mkdir(occupied_path, fs_ctx=fs._pathlock_fs_ctx(ctx, lease) if lease else None)

    if when == "before":
        await create()
    elif when == "after_acquisition":
        async def acquire_and_inject(*args, **kwargs):
            lease = await acquire(*args, **kwargs)
            await create(lease)
            return lease
        monkeypatch.setattr(agfs, "pathlock_acquire_batch", acquire_and_inject)
    else:
        # For backup exclusive creation an existing target is required.
        if which == "backup":
            target = enabled_tree[3]
        # Use the exact fs_ctx rather than reconstructing its opaque lease.
        async def mkdir_race(path, *args, **kwargs):
            nonlocal fired
            if path == occupied_path and not fired:
                fired = True
                await mkdir(path, *args, **kwargs)
            return await mkdir(path, *args, **kwargs)
        monkeypatch.setattr(agfs, "mkdir", mkdir_race)
    old_revision = await revision(fs, target, ctx) if which == "backup" and when == "exclusive_create" else "absent"
    with pytest.raises((ConflictError, OpenVikingError)):
        await tree_replace.replace_tree(
            fs, target_uri=target, staging_uri=staging, backup_uri=backup,
            expected_target_revision=old_revision, expected_staging_revision=staging_revision, ctx=ctx,
        )
    assert fired
    assert (await agfs.stat(occupied_path))["isDir"]
    assert await agfs.ls(occupied_path) == []
    assert await raw(fs, staging + "/one.md", ctx) == b"new one"


@pytest.mark.parametrize("change", ["recreate", "deleting"])
@pytest.mark.parametrize("operation", ["revision", "replace", "move", "absent_batch", "hash_batch", "rm"])
async def test_root_caller_rechecks_authoritative_account_after_final_acquisition(
    enabled_tree, monkeypatch, change, operation
):
    fs, _, ctx, target, staging, backup, _, _ = enabled_tree
    assert ctx.role == Role.ROOT
    agfs, manager = fs._async_agfs, fs._account_identity_manager
    old_revision = await revision(fs, target, ctx)
    stage_revision = await revision(fs, staging, ctx)
    original = agfs.pathlock_acquire_batch
    fired = False

    async def acquire_and_change(*args, **kwargs):
        nonlocal fired
        lease = await original(*args, **kwargs)
        if not fired:
            fired = True
            if change == "deleting":
                await manager.begin_deletion(
                    ctx.account_id, None, task_id="fixture-delete",
                    owner_account_id=ctx.account_id, owner_user_id=ctx.user.user_id,
                )
            else:
                # Replace the authoritative same-ID account creation identity,
                # not a fake identity-manager response or watcher cache.
                data = json.loads(await agfs.cat(ACCOUNTS_PATH))
                data["accounts"][ctx.account_id]["creation_id"] = str(uuid4())
                await agfs.write(ACCOUNTS_PATH, json.dumps(data).encode())
        return lease

    monkeypatch.setattr(agfs, "pathlock_acquire_batch", acquire_and_change)
    with pytest.raises((ConflictError, FailedPreconditionError)):
        if operation == "revision":
            await revision(fs, target, ctx)
        elif operation == "replace":
            await tree_replace.replace_tree(
                fs, target_uri=target, staging_uri=staging, backup_uri=backup,
                expected_target_revision=old_revision, expected_staging_revision=stage_revision, ctx=ctx,
            )
        elif operation == "move":
            await fs.mv(target, backup, ctx=ctx, no_replace=True)
        elif operation == "rm":
            await fs.rm(target + "/missing", ctx=ctx)
        else:
            precondition = ({"expected_absent": True} if operation == "absent_batch" else
                            {"expected_sha256": hashlib.sha256(b"old one").hexdigest()})
            await ContentWriteCoordinator(fs)._batch_write(
                root_uri=target, operations=[{"uri": target + "/one.md", "content": "changed", **precondition}],
                ctx=ctx, defer_processing=True,
            )
    assert fired
    assert await raw(fs, target + "/one.md", ctx) == b"old one"
    assert not await fs.exists(backup, ctx=ctx)


@pytest.mark.parametrize("operation", ["revision", "replace", "move", "batch"])
async def test_missing_deeper_parent_is_not_recreated(enabled_tree, monkeypatch, operation):
    fs, _, ctx, target, staging, backup, _, _ = enabled_tree
    parent = "viking://resources/published"
    path = fs._uri_to_path(parent, ctx=ctx)
    agfs, original = fs._async_agfs, fs._async_agfs.pathlock_acquire_batch
    old_revision, stage_revision = await revision(fs, target, ctx), await revision(fs, staging, ctx)
    fired = False

    async def remove_then_acquire(*args, **kwargs):
        nonlocal fired
        if not fired:
            fired = True
            await agfs.rm(path, recursive=True)
        return await original(*args, **kwargs)

    monkeypatch.setattr(agfs, "pathlock_acquire_batch", remove_then_acquire)
    with pytest.raises((NotFoundError, ConflictError, OpenVikingError)):
        if operation == "revision":
            await revision(fs, target, ctx)
        elif operation == "replace":
            await tree_replace.replace_tree(
                fs, target_uri=target, staging_uri=staging, backup_uri=backup,
                expected_target_revision=old_revision, expected_staging_revision=stage_revision, ctx=ctx,
            )
        elif operation == "move":
            await fs.mv(staging, target + '/new', ctx=ctx, no_replace=True)
        else:
            await ContentWriteCoordinator(fs)._batch_write(
                root_uri=target, operations=[{"uri": target + "/new.md", "content": "new", "expected_absent": True}],
                ctx=ctx, defer_processing=True,
            )
    assert fired
    assert not await fs.exists(parent, ctx=ctx)


async def test_namespace_root_recreation_is_accepted_but_revision_is_truthful(enabled_tree, monkeypatch):
    fs, _, ctx, target, *_ = enabled_tree
    root_path = fs._uri_to_path("viking://resources", ctx=ctx)
    agfs, original = fs._async_agfs, fs._async_agfs.pathlock_acquire_batch
    fired = False

    async def remove_then_acquire(*args, **kwargs):
        nonlocal fired
        if not fired:
            fired = True
            await agfs.rm(root_path, recursive=True)
        return await original(*args, **kwargs)

    monkeypatch.setattr(agfs, "pathlock_acquire_batch", remove_then_acquire)
    result = await tree_replace.tree_revision(fs, "viking://resources", ctx=ctx)
    assert fired and result["exists"] and result["revision"] != "absent"
    assert result["files"] == result["directories"] == 0


async def test_absent_resource_rm_vector_producer_conflicts_with_foreign_root(enabled_tree):
    fs, backend, ctx, target, *_ = enabled_tree
    missing = target + "/orphan.md"
    record = {"id": "orphan", "uri": missing, "level": 2, "account_id": ctx.account_id, "vector": [0.1] * 4}
    await backend.upsert(record, ctx=ctx)
    outer = await fs._async_agfs.pathlock_acquire_tree(fs._uri_to_path("viking://resources", ctx=ctx))
    try:
        with pytest.raises((LockAcquisitionError, ResourceBusyError)):
            await fs.rm(missing, ctx=ctx, auto_pathlock=False)
        assert await backend.get(["orphan"], ctx=ctx)
        assert not await fs.exists(missing, ctx=ctx)
    finally:
        await fs._async_agfs.pathlock_release(outer)
    await fs.rm(missing, ctx=ctx)
    assert not await backend.get(["orphan"], ctx=ctx)
    assert not await fs.exists(missing, ctx=ctx)


@pytest.mark.parametrize("change", ["gone", "file"])
async def test_move_rechecks_directory_source_without_recreating_it(enabled_tree, monkeypatch, change):
    fs, _, ctx, target, _, backup, _, _ = enabled_tree
    agfs, original = fs._async_agfs, fs._async_agfs.pathlock_acquire_batch
    source = fs._uri_to_path(target, ctx=ctx)
    fired = False

    async def change_then_acquire(*args, **kwargs):
        nonlocal fired
        if not fired:
            fired = True
            await agfs.rm(source, recursive=True)
            if change == "file":
                await agfs.write(source, b"foreign file")
        return await original(*args, **kwargs)

    monkeypatch.setattr(agfs, "pathlock_acquire_batch", change_then_acquire)
    with pytest.raises(ConflictError):
        await fs.mv(target, backup, ctx=ctx, no_replace=True, vector_mode="none")
    assert fired and not await fs.exists(backup, ctx=ctx)
    if change == "file":
        assert await raw(fs, target, ctx) == b"foreign file"
    else:
        assert not await fs.exists(target, ctx=ctx)


async def test_qualified_resources_fail_closed_without_identity_manager(enabled_tree):
    fs, _, ctx, target, *_ = enabled_tree
    fs.set_account_identity_manager(None)
    with pytest.raises(FailedPreconditionError):
        await revision(fs, target, ctx)
    with pytest.raises(FailedPreconditionError):
        await fs.rm(target + "/absent", ctx=ctx, auto_pathlock=False)


async def test_batch_preconditions_and_ordinary_create_defaults(enabled_tree, monkeypatch):
    from unittest.mock import AsyncMock
    fs, _, ctx, target, *_ = enabled_tree
    coordinator = ContentWriteCoordinator(fs)
    # Queue execution belongs to a separate gate; only deferred writes are used.
    monkeypatch.setattr(coordinator, "_enqueue_content_write_root", AsyncMock(return_value="fixture-task"))
    factory = AsyncMock()
    uri = target + "/one.md"
    with pytest.raises(ConflictError):
        await coordinator._batch_write(
            root_uri=target, operations=[{"uri": uri, "content": "blocked", "expected_absent": True}],
            ctx=ctx, defer_processing=True, root_task_factory=factory,
        )
    with pytest.raises(ConflictError):
        await coordinator._batch_write(
            root_uri=target, operations=[{"uri": uri, "content": "blocked", "expected_sha256": "0" * 64}],
            ctx=ctx, defer_processing=True, root_task_factory=factory,
        )
    assert await raw(fs, uri, ctx) == b"old one"
    result = await coordinator._batch_write(
        root_uri=target, operations=[{
            "uri": uri, "content": "hash-qualified", "mode": "replace",
            "expected_sha256": hashlib.sha256(b"old one").hexdigest(),
        }], ctx=ctx, defer_processing=True, root_task_factory=factory,
    )
    assert result["updated"] == [uri]
    assert await raw(fs, uri, ctx) == b"hash-qualified"
    # Ordinary create remains materialization/overwrite, without requiring D2.
    fs.set_account_identity_manager(None)
    result = await coordinator._batch_write(
        root_uri=target, operations=[{"uri": uri, "content": "ordinary", "mode": "create"}],
        ctx=ctx, defer_processing=True, root_task_factory=factory,
    )
    assert result["updated"] == [uri]
    assert await raw(fs, uri, ctx) == b"ordinary"


async def test_absent_target_install_failure_restores_absence_and_keeps_stage(enabled_tree):
    fs, _, ctx, _, staging, backup, _, _ = enabled_tree
    target = "viking://resources/published/absent"
    async def fail(phase):
        if phase == "install":
            raise OSError("injected install failure")
    with pytest.raises(OpenVikingError) as raised:
        await tree_replace.replace_tree(
            fs, target_uri=target, staging_uri=staging, backup_uri=backup,
            expected_target_revision="absent", expected_staging_revision=await revision(fs, staging, ctx),
            ctx=ctx, fault=fail,
        )
    assert raised.value.details["state"] == "rolled_back"
    assert raised.value.details["target_revision"] == "absent"
    assert not await fs.exists(target, ctx=ctx)
    assert not await fs.exists(backup, ctx=ctx)
    assert await raw(fs, staging + "/one.md", ctx) == b"new one"


async def test_resources_root_rm_is_fenced_and_next_revision_reports_materialized_root(enabled_tree):
    fs, _, ctx, target, *_ = enabled_tree
    root = "viking://resources"
    admitted = await fs._capture_resource_identity(ctx)
    lease = await fs._acquire_resource_coverage(ctx, admitted)
    try:
        with pytest.raises((LockAcquisitionError, ResourceBusyError)):
            await fs.rm(root, recursive=True, ctx=ctx)
    finally:
        await fs._async_agfs.pathlock_release(lease)
    await fs.rm(root, recursive=True, ctx=ctx)
    result = await tree_replace.tree_revision(fs, root, ctx=ctx)
    assert result["exists"] and result["revision"] != "absent"
    assert result["files"] == result["directories"] == 0
    assert not await fs.exists(target, ctx=ctx)


async def test_existing_empty_target_can_be_replaced_with_its_exact_revision(enabled_tree):
    from openviking.storage.vector_ids import vector_record_id
    fs, backend, ctx, _, staging, backup, _, _ = enabled_tree
    target = "viking://resources/published/empty"
    await fs.mkdir(target, ctx=ctx)
    await backend.upsert({
        "id": vector_record_id(ctx.account_id, target, 0), "uri": target, "level": 0,
        "account_id": ctx.account_id, "vector": [0.1] * 4, "abstract": "",
    }, ctx=ctx)
    before = await revision(fs, target, ctx)
    assert before != "absent"
    result = await tree_replace.replace_tree(
        fs, target_uri=target, staging_uri=staging, backup_uri=backup,
        expected_target_revision=before, expected_staging_revision=await revision(fs, staging, ctx), ctx=ctx,
    )
    assert result["content_committed"] and result["backup_retained"]
    assert await fs.exists(backup, ctx=ctx)
    assert await raw(fs, target + "/one.md", ctx) == b"new one"


async def test_service_move_covers_version_bookkeeping_and_extends_owner(enabled_tree, monkeypatch):
    from unittest.mock import AsyncMock
    from openviking.service.fs_service import FSService
    fs, _, ctx, target, _, backup, _, _ = enabled_tree
    service = FSService(viking_fs=fs)
    monkeypatch.setattr(service, "_refresh_move_parents", AsyncMock())
    events = []
    async def protected(phase):
        with pytest.raises((LockAcquisitionError, ResourceBusyError)):
            await fs._async_agfs.write(
                fs._uri_to_path("viking://resources/published/unrelated.md", ctx=ctx), b"foreign"
            )
        events.append(phase)
    class Versions:
        async def record_operation(self, uris, context, name, options, operation, **kwargs):
            await protected("record_operation")
            return await operation()
        async def guard_transfer(self, *args, **kwargs):
            await protected("guard_transfer")
            return True
        async def before_move(self, *args):
            await protected("before_move")
        async def register_transfer(self, *args, **kwargs):
            await protected("register_transfer")
            assert kwargs["lease_ref"] is not None
    service.compile_versions = Versions()
    await service.mv(target, backup, ctx=ctx, no_replace=True)
    assert events == ["record_operation", "guard_transfer", "before_move", "register_transfer"]
    assert not await fs.exists(target, ctx=ctx)
    assert await raw(fs, backup + "/one.md", ctx) == b"old one"
    await fs.write_file_bytes("viking://resources/published/after.md", b"after", ctx=ctx)


async def test_resource_rm_rechecks_type_after_acquisition(enabled_tree, monkeypatch):
    from openviking_cli.exceptions import InvalidArgumentError
    fs, _, ctx, target, *_ = enabled_tree
    uri = target + "/one.md"
    agfs, original = fs._async_agfs, fs._async_agfs.pathlock_acquire_batch
    path = fs._uri_to_path(uri, ctx=ctx)
    fired = False
    async def change_then_acquire(*args, **kwargs):
        nonlocal fired
        if not fired:
            fired = True
            await agfs.rm(path)
            await agfs.mkdir(path)
            await agfs.write(path + "/foreign.md", b"foreign")
        return await original(*args, **kwargs)
    monkeypatch.setattr(agfs, "pathlock_acquire_batch", change_then_acquire)
    with pytest.raises(InvalidArgumentError):
        await fs.rm(uri, recursive=False, ctx=ctx)
    assert fired
    assert await raw(fs, uri + "/foreign.md", ctx) == b"foreign"


@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize("when", ["before", "after_acquisition"])
async def test_move_never_adopts_or_removes_foreign_empty_destination(enabled_tree, monkeypatch, directory, when):
    fs, _, ctx, target, _, backup, _, _ = enabled_tree
    source = target if directory else target + "/one.md"
    agfs = fs._async_agfs
    path = fs._uri_to_path(backup, ctx=ctx)
    original = agfs.pathlock_acquire_batch
    fired = False
    if when == "before":
        await agfs.mkdir(path)
        fired = True
    else:
        async def acquire_and_inject(*args, **kwargs):
            nonlocal fired
            lease = await original(*args, **kwargs)
            if not fired:
                fired = True
                await agfs.mkdir(path, fs_ctx=fs._pathlock_fs_ctx(ctx, lease))
            return lease
        monkeypatch.setattr(agfs, "pathlock_acquire_batch", acquire_and_inject)
    with pytest.raises(ConflictError):
        await fs.mv(source, backup, ctx=ctx, no_replace=True)
    assert fired and (await agfs.stat(path))["isDir"]
    assert await agfs.ls(path) == []
    assert await raw(fs, target + "/one.md", ctx) == b"old one"


async def test_file_move_copy_failure_never_cleans_foreign_empty_destination(enabled_tree, monkeypatch):
    fs, _, ctx, target, _, backup, _, _ = enabled_tree
    agfs, original = fs._async_agfs, fs._async_agfs.cp
    path = fs._uri_to_path(backup, ctx=ctx)
    async def inject_then_copy(*args, **kwargs):
        await agfs.mkdir(path, fs_ctx=kwargs["fs_ctx"])
        return await original(*args, **kwargs)
    monkeypatch.setattr(agfs, "cp", inject_then_copy)
    with pytest.raises(Exception):
        await fs.mv(target + "/one.md", backup, ctx=ctx, no_replace=True)
    assert (await agfs.stat(path))["isDir"]
    assert await agfs.ls(path) == []
    assert await raw(fs, target + "/one.md", ctx) == b"old one"


@pytest.mark.parametrize("remove_parent", [False, True])
async def test_mixed_namespace_move_uses_stable_account_ancestor(enabled_tree, monkeypatch, remove_parent):
    fs, _, ctx, target, *_ = enabled_tree
    destination_parent = f"viking://user/{ctx.user.user_id}/resources/moved"
    destination = destination_parent + "/one.md"
    await fs.mkdir(destination_parent, ctx=ctx)
    agfs, original = fs._async_agfs, fs._async_agfs.pathlock_acquire_batch
    parent_path = fs._uri_to_path(destination_parent, ctx=ctx)
    account_root = fs._uri_to_path("viking://resources", ctx=ctx).rsplit("/", 1)[0]
    requests_seen = []
    async def acquire(requests, *args, **kwargs):
        if not requests_seen:
            requests_seen.extend(requests)
            if remove_parent:
                await agfs.rm(parent_path, recursive=True)
        return await original(requests, *args, **kwargs)
    monkeypatch.setattr(agfs, "pathlock_acquire_batch", acquire)
    if remove_parent:
        with pytest.raises(NotFoundError):
            await fs.mv(target + "/one.md", destination, ctx=ctx, no_replace=True)
        assert not await fs.exists(destination_parent, ctx=ctx)
        assert await raw(fs, target + "/one.md", ctx) == b"old one"
    else:
        await fs.mv(target + "/one.md", destination, ctx=ctx, no_replace=True)
        assert await raw(fs, destination, ctx) == b"old one"
    assert requests_seen == [{"path": account_root, "kind": "tree"}]
