# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Watch eligibility gate for the qualified resources mutators.

Real RAGFS, real local stores, enabled native ACLs and the resources-root
lease. Only the watch scheduler shell and the ingestion writer are synthetic:
the WatchManager, UriMutationCoordinator and TaskTracker are the real ones.
"""

import asyncio
import hashlib
import threading
from unittest.mock import AsyncMock

import pytest

from openviking.connector.delegate import ConnectorDelegate
from openviking.resource.uri_mutation_coordinator import UriMutationCoordinator
from openviking.resource.watch_manager import WatchManager, WatchTask
from openviking.resource.watch_storage import (
    WATCH_TASK_STORAGE_BAK_URI,
    WATCH_TASK_STORAGE_TMP_URI,
    WATCH_TASK_STORAGE_URI,
)
from openviking.server.identity import RequestContext, Role
from openviking.service.fs_service import FSService
from openviking.service.task_store import PersistentTaskStore
from openviking.service.task_tracker import TaskTracker
from openviking.service.task_tracker_concurrency import run_to_completion
from openviking.service.tree_replace import tree_revision
from openviking.storage.acl import AclMode
from openviking_cli.exceptions import OpenVikingError
from openviking_cli.session.user_id import UserIdentifier
from tests.storage.test_resource_maintenance import seed as seed_orphan
from tests.storage.test_tree_replace_structured import (
    binding_fs as binding_fs,
)
from tests.storage.test_tree_replace_structured import (
    enabled_tree as _enabled_tree,
)
from tests.storage.test_tree_replace_structured import (
    indexed_fs as indexed_fs,
)
from tests.storage.test_tree_replace_structured import raw

enabled_tree = _enabled_tree

MOVED = "viking://resources/moved"
CALL_SITES = ["replace", "remove", "prune", "batch", "mv"]


class _Scheduler:
    """Synthetic scheduler shell: exposes only the real WatchManager."""

    def __init__(self, manager):
        self.watch_manager = manager


async def _harness(case, monkeypatch, watches):
    fs, backend, ctx, target, staging, backup, *_ = case
    coordinator = UriMutationCoordinator()
    manager = WatchManager(viking_fs=fs, uri_mutation_coordinator=coordinator)
    await manager.initialize()
    for watch in watches:
        manager._tasks[watch.task_id] = watch
    await manager._save_tasks()
    tracker = TaskTracker(PersistentTaskStore(fs._async_agfs))
    monkeypatch.setattr("openviking.service.task_tracker._instance", tracker)
    service = FSService(
        fs, watch_scheduler=_Scheduler(manager), uri_mutation_coordinator=coordinator
    )
    # Post-commit refresh producers are not the subject here.
    monkeypatch.setattr(service, "_refresh_move_parents", AsyncMock())
    monkeypatch.setattr(service, "_enqueue_transfer_rebuild", AsyncMock())
    watch_bytes = await raw(fs, WATCH_TASK_STORAGE_URI, ctx)
    return service, manager, tracker, coordinator, watch_bytes


def _primary(site, target):
    return target + "/orphan" if site == "prune" else target


async def _call(site, service, case, ctx=None):
    fs, backend, owner_ctx, target, staging, backup, *_ = case
    ctx = ctx or owner_ctx
    if site == "replace":
        return await service.replace_tree(
            target_uri=target, staging_uri=staging, backup_uri=backup,
            expected_target_revision=(await tree_revision(fs, target, ctx=owner_ctx))["revision"],
            expected_staging_revision=(await tree_revision(fs, staging, ctx=owner_ctx))["revision"],
            ctx=ctx,
        )
    if site == "remove":
        return await service.remove_resource_only(target, ctx, recursive=True)
    if site == "prune":
        return await service.prune_resource_index(target + "/orphan", ctx)
    if site == "batch":
        current = await raw(fs, target + "/one.md", owner_ctx)
        return await service.batch_write(
            root_uri=target,
            operations=[{
                "uri": target + "/one.md", "content": "gated", "mode": "replace",
                "expected_sha256": hashlib.sha256(current).hexdigest(),
            }],
            ctx=ctx, wait=False, defer_processing=True, edit_source="compile",
        )
    return await service.mv(target, MOVED, ctx, no_replace=True)


async def _assert_untouched(site, case, watch_bytes):
    fs, backend, ctx, target, *_ = case
    assert await raw(fs, WATCH_TASK_STORAGE_URI, ctx) == watch_bytes
    assert await raw(fs, target + "/one.md", ctx) == b"old one"
    assert not await fs.exists(MOVED, ctx=ctx)
    if site == "prune":
        assert await backend.get(["orphan"], ctx=ctx)


def _watch(ctx, to_uri, **fields):
    return WatchTask(
        path="https://example.invalid/source", to_uri=to_uri,
        account_id=fields.pop("account_id", ctx.account_id),
        user_id=fields.pop("user_id", "someone-else"), **fields,
    )


def _variants(ctx, primary):
    parent = primary.rsplit("/", 1)[0]
    connector = {"provider": ConnectorDelegate._WATCH_AUTH_PROVIDER}
    return {
        "equal": _watch(ctx, primary),
        "ancestor": _watch(ctx, parent),
        "descendant": _watch(ctx, primary + "/branch"),
        "paused": _watch(ctx, primary, is_active=False),
        "connector": _watch(ctx, primary, auth_state=connector, connector_states={}),
        "no_target_overlapping_parent": _watch(ctx, None, parent_uri=parent),
        "no_target_unknown_parent": _watch(ctx, None),
    }


@pytest.mark.parametrize("site", CALL_SITES)
@pytest.mark.parametrize(
    "variant",
    ["equal", "ancestor", "descendant", "paused", "connector",
     "no_target_overlapping_parent", "no_target_unknown_parent"],
)
async def test_overlapping_watch_refuses_before_any_effect(enabled_tree, monkeypatch, site, variant):
    fs, backend, ctx, target, *_ = enabled_tree
    if site == "prune":
        await seed_orphan(backend, target + "/orphan", ctx)
    watch = _variants(ctx, _primary(site, target))[variant]
    service, manager, _, _, watch_bytes = await _harness(enabled_tree, monkeypatch, [watch])
    before = watch.model_dump()
    with pytest.raises(OpenVikingError) as error:
        await _call(site, service, enabled_tree)
    assert error.value.code == "CONFLICT"
    # Generic: no task ID, owner or source path.
    message = str(error.value) + repr(getattr(error.value, "details", None))
    for secret in (watch.task_id, "someone-else", "example.invalid"):
        assert secret not in message
    await _assert_untouched(site, enabled_tree, watch_bytes)
    assert manager._tasks[watch.task_id].model_dump() == before


@pytest.mark.parametrize("site", CALL_SITES)
@pytest.mark.parametrize("variant", ["prefix_sibling", "other_account"])
async def test_unrelated_watch_does_not_gate(enabled_tree, monkeypatch, site, variant):
    fs, backend, ctx, target, *_ = enabled_tree
    if site == "prune":
        await seed_orphan(backend, target + "/orphan", ctx)
    primary = _primary(site, target)
    watch = (
        # A string-prefix sibling must not count as a path-boundary overlap.
        _watch(ctx, target + "s")
        if variant == "prefix_sibling"
        else _watch(ctx, primary, account_id="other-account")
    )
    service, manager, _, _, watch_bytes = await _harness(enabled_tree, monkeypatch, [watch])
    await _call(site, service, enabled_tree)
    # The qualified path never writes watch state, even on success.
    assert await raw(fs, WATCH_TASK_STORAGE_URI, ctx) == watch_bytes
    if site == "remove":
        assert not await fs.exists(target, ctx=ctx)
    elif site == "prune":
        assert not await backend.get(["orphan"], ctx=ctx)
    elif site == "batch":
        assert await raw(fs, target + "/one.md", ctx) == b"gated"
    elif site == "mv":
        assert await fs.exists(MOVED, ctx=ctx) and not await fs.exists(target, ctx=ctx)
    else:
        assert await raw(fs, target + "/one.md", ctx) == b"new one"


@pytest.mark.parametrize("site", CALL_SITES)
async def test_permission_denial_precedes_watch_conflict(enabled_tree, monkeypatch, site):
    fs, backend, ctx, target, *_ = enabled_tree
    if site == "prune":
        await seed_orphan(backend, target + "/orphan", ctx)
    user = RequestContext(user=UserIdentifier(ctx.account_id, "outsider"), role=Role.USER)
    owner = {"principal": "user:" + ctx.user.user_id, "level": "manage"}
    await fs.set_acl(target, [owner], ctx=ctx, acl_mode=AclMode.RESTRICTED)
    if site == "prune":
        await fs.acl_manager.set_acl(
            target + "/orphan", [owner], ctx, acl_mode=AclMode.RESTRICTED
        )
    watch = _watch(ctx, _primary(site, target))
    service, _, _, _, watch_bytes = await _harness(enabled_tree, monkeypatch, [watch])
    with pytest.raises(OpenVikingError) as error:
        await _call(site, service, enabled_tree, ctx=user)
    assert error.value.code == "PERMISSION_DENIED"
    await _assert_untouched(site, enabled_tree, watch_bytes)


@pytest.mark.parametrize(
    "uri", [WATCH_TASK_STORAGE_URI, WATCH_TASK_STORAGE_BAK_URI, WATCH_TASK_STORAGE_TMP_URI]
)
@pytest.mark.parametrize("operation", ["remove", "prune"])
async def test_watch_control_state_is_never_maintained(enabled_tree, monkeypatch, uri, operation):
    fs, backend, ctx, *_ = enabled_tree
    service, _, _, _, watch_bytes = await _harness(enabled_tree, monkeypatch, [])
    with pytest.raises(OpenVikingError) as error:
        if operation == "remove":
            await service.remove_resource_only(uri, ctx)
        else:
            await service.prune_resource_index(uri, ctx)
    assert error.value.code == "PERMISSION_DENIED"
    assert await raw(fs, WATCH_TASK_STORAGE_URI, ctx) == watch_bytes


@pytest.mark.parametrize("site", ["replace", "remove", "batch", "mv"])
async def test_executing_tick_delays_mutation_which_is_then_refused(
    enabled_tree, monkeypatch, site
):
    fs, backend, ctx, target, *_ = enabled_tree
    watch = _watch(ctx, target)
    service, _, _, coordinator, watch_bytes = await _harness(
        enabled_tree, monkeypatch, [watch]
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def tick():
        # The same access lease WatchScheduler._execute_task holds.
        async with coordinator.access(ctx.account_id, [target]):
            entered.set()
            await release.wait()

    ticking = asyncio.create_task(tick())
    await entered.wait()
    mutation = asyncio.create_task(_call(site, service, enabled_tree))
    await asyncio.sleep(0.3)
    assert not mutation.done()
    await _assert_untouched(site, enabled_tree, watch_bytes)
    release.set()
    await ticking
    with pytest.raises(OpenVikingError) as error:
        await mutation
    assert error.value.code == "CONFLICT"
    await _assert_untouched(site, enabled_tree, watch_bytes)


@pytest.mark.parametrize("site", CALL_SITES)
async def test_cancelled_ingestion_gates_until_its_write_settles(enabled_tree, monkeypatch, site):
    """R2: a timed-out tick cancels ingestion and the watch is then deleted.

    TaskTracker.cancel only requests cancellation. The writer is blocked inside
    a native-style thread call that run_to_completion cannot interrupt, so the
    Task stays ``cancelling`` until that write returns, and the gate refuses
    until then even though no watch remains.
    """
    fs, backend, ctx, target, *_ = enabled_tree
    if site == "prune":
        await seed_orphan(backend, target + "/orphan", ctx)
    service, manager, tracker, _, watch_bytes = await _harness(enabled_tree, monkeypatch, [])
    owner = "watch-owner"
    task = await tracker.create(
        "add_resource", resource_id=target, account_id=ctx.account_id, user_id=owner
    )
    await tracker.start(task.task_id, account_id=ctx.account_id, user_id=owner)
    in_write = threading.Event()
    finish_write = threading.Event()
    write_returned = threading.Event()

    def blocked_native_write():
        in_write.set()
        assert finish_write.wait(30)
        write_returned.set()

    async def ingestion():
        await tracker.register_running_task(
            task.task_id, account_id=ctx.account_id, user_id=owner
        )
        try:
            await run_to_completion(lambda: asyncio.to_thread(blocked_native_write))
        finally:
            await tracker.unregister_running_task(task.task_id)

    writer = asyncio.create_task(ingestion())
    await asyncio.to_thread(in_write.wait, 30)
    cancelled = await tracker.cancel(task.task_id, account_id=ctx.account_id, user_id=owner)
    # cancel returned while the write is still in flight.
    assert not write_returned.is_set()
    assert cancelled is not None and cancelled.status.value == "cancelling"
    with pytest.raises(OpenVikingError) as error:
        await _call(site, service, enabled_tree)
    assert error.value.code == "CONFLICT"
    await _assert_untouched(site, enabled_tree, watch_bytes)
    finish_write.set()
    with pytest.raises(asyncio.CancelledError):
        await writer
    assert write_returned.is_set()
    settled = await tracker.get(task.task_id, account_id=ctx.account_id, user_id=owner)
    assert settled.status.value == "cancelled"
    await _call(site, service, enabled_tree)


async def test_unfinished_ingestion_without_known_target_fails_closed(enabled_tree, monkeypatch):
    fs, backend, ctx, target, *_ = enabled_tree
    service, _, tracker, _, watch_bytes = await _harness(enabled_tree, monkeypatch, [])
    await tracker.create("add_resource", account_id=ctx.account_id, user_id="watch-owner")
    with pytest.raises(OpenVikingError) as error:
        await _call("remove", service, enabled_tree)
    assert error.value.code == "CONFLICT"
    await _assert_untouched("remove", enabled_tree, watch_bytes)


async def test_initializing_watch_manager_fails_closed(enabled_tree, monkeypatch):
    fs, backend, ctx, target, *_ = enabled_tree
    service, manager, _, _, watch_bytes = await _harness(enabled_tree, monkeypatch, [])
    manager._initialized = False
    with pytest.raises(OpenVikingError) as error:
        await _call("remove", service, enabled_tree)
    assert error.value.code == "FAILED_PRECONDITION"
    await _assert_untouched("remove", enabled_tree, watch_bytes)


async def test_disabled_scheduler_allows(enabled_tree, monkeypatch):
    fs, backend, ctx, target, *_ = enabled_tree
    service, *_ = await _harness(enabled_tree, monkeypatch, [])
    service._watch_scheduler = _Scheduler(None)
    await _call("remove", service, enabled_tree)
    assert not await fs.exists(target, ctx=ctx)


async def test_ordinary_mv_keeps_watch_sync_and_is_not_gated(enabled_tree, monkeypatch):
    fs, backend, ctx, target, *_ = enabled_tree
    watch = _watch(ctx, target)
    service, manager, *_ = await _harness(enabled_tree, monkeypatch, [watch])
    gate = AsyncMock(side_effect=AssertionError("ordinary mv gated"))
    monkeypatch.setattr(service, "_refuse_watched_resource_targets", gate)
    await service.mv(target, MOVED, ctx)
    assert manager._tasks[watch.task_id].to_uri == MOVED
    gate.assert_not_called()
