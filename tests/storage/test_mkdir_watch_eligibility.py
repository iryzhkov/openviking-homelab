# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Mkdir uses the existing watch gate before changing any directory metadata."""

from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.acl import AclMode
from openviking_cli.exceptions import OpenVikingError
from openviking_cli.session.user_id import UserIdentifier
from tests.storage.test_watch_eligibility_gate import (
    _assert_untouched,
    _harness,
    _variants,
    _watch,
)
from tests.storage.test_watch_eligibility_gate import (
    binding_fs as binding_fs,
)
from tests.storage.test_watch_eligibility_gate import (
    enabled_tree as enabled_tree,
)
from tests.storage.test_watch_eligibility_gate import (
    indexed_fs as indexed_fs,
)


@pytest.mark.parametrize(
    "variant",
    [
        "equal",
        "ancestor",
        "descendant",
        "paused",
        "connector",
        "no_target_overlapping_parent",
        "no_target_unknown_parent",
    ],
)
async def test_mkdir_watch_overlap_refuses_before_creating_target(
    enabled_tree, monkeypatch, variant
):
    fs, _, ctx, target, *_ = enabled_tree
    directory = target + "/new-dir"
    watch = _variants(ctx, directory)[variant]
    service, manager, _, _, watch_bytes = await _harness(enabled_tree, monkeypatch, [watch])
    before = watch.model_dump()
    with pytest.raises(OpenVikingError) as error:
        await service.mkdir(directory, ctx, description="must not be written")
    assert error.value.code == "CONFLICT"
    assert not await fs.exists(directory, ctx=ctx)
    await _assert_untouched("mkdir", enabled_tree, watch_bytes)
    assert manager._tasks[watch.task_id].model_dump() == before


@pytest.mark.parametrize("variant", ["prefix_sibling", "other_account"])
async def test_mkdir_unrelated_watch_preserves_creation(enabled_tree, monkeypatch, variant):
    fs, _, ctx, target, *_ = enabled_tree
    directory = target + "/new-dir"
    watch = (
        _watch(ctx, directory + "s")
        if variant == "prefix_sibling"
        else _watch(ctx, directory, account_id="other-account")
    )
    service, _, _, _, watch_bytes = await _harness(enabled_tree, monkeypatch, [watch])
    # Only post-write embedding is outside this gate test. Native mkdir,
    # generated abstract, ACLs, leases and watch state are real.
    monkeypatch.setattr("openviking.service.fs_service.vectorize_directory_meta", AsyncMock())
    await service.mkdir(directory, ctx, description="created")
    assert await fs.exists(directory, ctx=ctx)
    assert await fs.exists(directory + "/.abstract.md", ctx=ctx)
    await _assert_untouched("mkdir", enabled_tree, watch_bytes)


async def test_mkdir_acl_denial_precedes_watch_disclosure(enabled_tree, monkeypatch):
    fs, _, ctx, target, *_ = enabled_tree
    directory = target + "/new-dir"
    owner = {"principal": "user:" + ctx.user.user_id, "level": "manage"}
    await fs.set_acl(target, [owner], ctx=ctx, acl_mode=AclMode.RESTRICTED)
    service, _, _, _, watch_bytes = await _harness(
        enabled_tree, monkeypatch, [_watch(ctx, directory)]
    )
    outsider = RequestContext(user=UserIdentifier(ctx.account_id, "outsider"), role=Role.USER)
    with pytest.raises(OpenVikingError) as error:
        await service.mkdir(directory, outsider)
    assert error.value.code == "PERMISSION_DENIED"
    assert not await fs.exists(directory, ctx=ctx)
    await _assert_untouched("mkdir", enabled_tree, watch_bytes)


async def test_mkdir_pending_ingestion_refuses_without_watch_record(enabled_tree, monkeypatch):
    fs, _, ctx, target, *_ = enabled_tree
    directory = target + "/new-dir"
    service, _, tracker, _, watch_bytes = await _harness(enabled_tree, monkeypatch, [])
    await tracker.create(
        "add_resource", resource_id=target, account_id=ctx.account_id, user_id="watch-owner"
    )
    with pytest.raises(OpenVikingError) as error:
        await service.mkdir(directory, ctx)
    assert error.value.code == "CONFLICT"
    assert not await fs.exists(directory, ctx=ctx)
    await _assert_untouched("mkdir", enabled_tree, watch_bytes)


async def test_mkdir_uninitialized_watch_manager_refuses(enabled_tree, monkeypatch):
    fs, _, ctx, target, *_ = enabled_tree
    directory = target + "/new-dir"
    service, manager, _, _, watch_bytes = await _harness(enabled_tree, monkeypatch, [])
    manager._initialized = False
    with pytest.raises(OpenVikingError) as error:
        await service.mkdir(directory, ctx)
    assert error.value.code == "FAILED_PRECONDITION"
    assert not await fs.exists(directory, ctx=ctx)
    await _assert_untouched("mkdir", enabled_tree, watch_bytes)
