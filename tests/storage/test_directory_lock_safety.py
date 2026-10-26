# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Directory-lock error propagation and conservative recovery observations."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from openviking.service import tree_replace
from openviking.storage.errors import LockAcquisitionError, ResourceBusyError
from openviking.storage.viking_fs._ops import _OpsMixin
from openviking_cli.exceptions import OpenVikingError


class ParentOps(_OpsMixin):
    def __init__(self, agfs):
        self._async_agfs = agfs

    def _pathlock_fs_ctx(self, ctx, lease):
        return {"lease_ref": "test-lease"}


@pytest.mark.parametrize("error_type", [LockAcquisitionError, ResourceBusyError])
async def test_ensure_parents_does_not_retry_lock_refusal(error_type):
    error = error_type("parent is busy")
    agfs = SimpleNamespace(
        ensure_parent_dirs=AsyncMock(side_effect=error),
        mkdir=AsyncMock(),
    )
    with pytest.raises(error_type) as raised:
        await ParentOps(agfs)._ensure_parent_dirs("/local/account/resources/new/file")
    assert raised.value is error
    agfs.mkdir.assert_not_awaited()


@pytest.mark.parametrize("error_type", [LockAcquisitionError, ResourceBusyError, OSError])
async def test_parent_fallback_propagates_failure(error_type):
    error = error_type("cannot create parent")
    agfs = SimpleNamespace(mkdir=AsyncMock(side_effect=error))
    with pytest.raises(error_type) as raised:
        await ParentOps(agfs)._mkdir_path_with_parents("/local/account/resources/new")
    assert raised.value is error
    assert agfs.mkdir.await_count == 1


@pytest.mark.parametrize("message", ["lock lease does not exist", "another lease already owns path"])
async def test_parent_fallback_does_not_infer_existence_from_lock_error(message):
    agfs = SimpleNamespace(mkdir=AsyncMock(side_effect=RuntimeError(message)))
    with pytest.raises(RuntimeError, match=message):
        await ParentOps(agfs)._mkdir_path_with_parents("/local/account")
    assert agfs.mkdir.await_count == 1


async def test_parent_fallback_keeps_existing_directory_default():
    agfs = SimpleNamespace(mkdir=AsyncMock(side_effect=[FileExistsError("already exists"), None]))
    await ParentOps(agfs)._mkdir_path_with_parents("/local/account")
    assert agfs.mkdir.await_count == 2


async def test_safe_read_distinguishes_absent_empty_and_unknown():
    read = AsyncMock(side_effect=[None, [], OSError("read unavailable")])
    absent = await tree_replace._safe_read(read, "/absent")
    empty = await tree_replace._safe_read(read, "/empty")
    unknown = await tree_replace._safe_read(read, "/unreadable")
    assert tree_replace._reported_revision(absent) == "absent"
    assert tree_replace._reported_revision(empty).startswith("tree-sha256:")
    assert tree_replace._reported_revision(unknown) == "unknown"


async def test_recovery_unknown_target_never_cleans_foreign_empty_backup():
    async def read(path):
        if path == "/backup":
            return []
        raise OSError("observation unavailable")

    agfs = SimpleNamespace(rm=AsyncMock())
    viking_fs = Mock(
        spec=tree_replace.VikingFS,
        _async_agfs=agfs,
        _update_vector_store_uris=AsyncMock(),
    )
    state = {
        "target_uri": "viking://resources/target",
        "staging_uri": "viking://resources/staging",
        "backup_uri": "viking://resources/backup",
        "previous_revision": "absent",
    }
    with pytest.raises(OpenVikingError) as raised:
        await tree_replace._recover(
            viking_fs,
            state=state,
            failure=OSError("install failed"),
            backup_verified=False,
            target_entries=None,
            read=read,
            paths=("/target", "/staging", "/backup"),
            ctx=Mock(spec=tree_replace.RequestContext),
            lease={},
            fs_ctx={},
        )
    details = raised.value.details
    assert details["state"] == "rollback_unavailable"
    assert details["observation_failure"] == "target_unreadable"
    assert details["target_revision"] == "unknown"
    assert {item["uri"]: item["revision"] for item in details["retained"]} == {
        state["backup_uri"]: tree_replace.revision_of([]),
        state["staging_uri"]: "unknown",
    }
    agfs.rm.assert_not_awaited()
    viking_fs._update_vector_store_uris.assert_not_awaited()
