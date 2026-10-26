"""Committed replacement outcomes survive cleanup observation failures."""

import pytest

from openviking.service import tree_replace
from tests.storage.test_tree_replace_structured import (
    binding_fs as binding_fs,
)
from tests.storage.test_tree_replace_structured import (
    enabled_tree as enabled_tree,
)
from tests.storage.test_tree_replace_structured import (
    indexed_fs as indexed_fs,
)
from tests.storage.test_tree_replace_structured import (
    replace_case,
)


@pytest.mark.parametrize("unreadable", ["staging", "backup", "both"])
async def test_committed_cleanup_reports_unreadable_copies(enabled_tree, monkeypatch, unreadable):
    fs, _, root, target, staging, backup, *_ = enabled_tree
    original = tree_replace._tree_entries
    started = False
    paths = {
        "staging": fs._uri_to_path(staging, ctx=root),
        "backup": fs._uri_to_path(backup, ctx=root),
    }

    async def observe(viking_fs, path, *args, **kwargs):
        selected = set(paths.values()) if unreadable == "both" else {paths[unreadable]}
        if started and path in selected:
            raise OSError("cleanup observations unavailable")
        return await original(viking_fs, path, *args, **kwargs)

    async def fault(phase):
        nonlocal started
        if phase == "cleanup":
            started = True
            raise OSError("post-install cleanup unavailable")

    monkeypatch.setattr(tree_replace, "_tree_entries", observe)
    result = await replace_case(enabled_tree, fault=fault, retain_backup=False)
    assert result["content_committed"] is True
    assert result["index_status"] == "transferred"
    assert result["cleanup"]["status"] == "failed"
    assert set(result["cleanup"]["retained"]) == {staging, backup}
    assert result["backup_retained"] is True
    selected = (
        {staging, backup}
        if unreadable == "both"
        else {staging if unreadable == "staging" else backup}
    )
    for uri in selected:
        assert result["cleanup"]["retained_revisions"][uri] == "unknown"
    assert await fs.read_file_bytes(target + "/one.md", ctx=root) == b"new one"


@pytest.mark.parametrize("copy", ["staging", "backup"])
@pytest.mark.parametrize("state", ["empty", "absent"])
async def test_committed_cleanup_distinguishes_empty_from_absent(
    enabled_tree, monkeypatch, copy, state
):
    fs, _, root, target, staging, backup, *_ = enabled_tree
    selected = staging if copy == "staging" else backup
    path = fs._uri_to_path(selected, ctx=root)
    original_clear = tree_replace.clear_skill_package_contents
    owner = []

    async def clear(viking_fs, uri, **kwargs):
        owner.append(kwargs["lease_ref"])
        return await original_clear(viking_fs, uri, **kwargs)

    async def fault(phase):
        if phase != "cleanup":
            return
        assert owner
        # Exercise a real empty/absent native tree, not a synthetic read result.
        await original_clear(fs, selected, ctx=root, lease_ref=owner[-1])
        entries = await tree_replace._tree_entries(
            fs, path, fs._pathlock_fs_ctx(root, owner[-1]), ctx=root
        )
        assert entries == []
        if state == "absent":
            await fs._async_agfs.rm(
                path, recursive=True, fs_ctx=fs._pathlock_fs_ctx(root, owner[-1])
            )
        raise OSError("cleanup stopped after partial progress")

    monkeypatch.setattr(tree_replace, "clear_skill_package_contents", clear)
    result = await replace_case(enabled_tree, fault=fault, retain_backup=False)
    assert result["content_committed"] is True
    assert result["cleanup"]["status"] == "failed"
    assert (selected in result["cleanup"]["retained"]) is (state == "empty")
    if state == "empty":
        assert result["cleanup"]["retained_revisions"][selected] == tree_replace.revision_of([])
    else:
        assert selected not in result["cleanup"].get("retained_revisions", {})
    if copy == "backup":
        assert result["backup_retained"] is (state == "empty")
    assert await fs.read_file_bytes(target + "/one.md", ctx=root) == b"new one"
