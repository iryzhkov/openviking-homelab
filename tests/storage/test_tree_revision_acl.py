"""Native ACL regressions for non-disclosing tree revisions and preconditions."""

import json

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.service import tree_replace
from openviking.service.fs_service import FSService
from openviking.storage.acl import AclAction, AclMode
from openviking.storage.vector_ids import vector_record_id
from openviking_cli.exceptions import PermissionDeniedError
from openviking_cli.session.user_id import UserIdentifier
from tests.storage.test_tree_replace_structured import (
    binding_fs as binding_fs,
)
from tests.storage.test_tree_replace_structured import (
    enabled_tree as enabled_tree,
)
from tests.storage.test_tree_replace_structured import (
    indexed_fs as indexed_fs,
)


async def _actor(case, *, level="manage"):
    fs, backend, root, target, staging, backup, *_ = case
    actor = RequestContext(user=UserIdentifier(root.account_id, "tree-acl-actor"), role=Role.USER)
    owner = {"principal": "user:" + root.user.user_id, "level": "manage"}
    grant = {"principal": "user:" + actor.user.user_id, "level": level}
    for uri in (target, staging):
        await fs.set_acl(uri, [owner, grant], ctx=root, acl_mode=AclMode.RESTRICTED)
    archive = backup.rsplit("/", 1)[0]
    await backend.upsert(
        {
            "id": vector_record_id(root.account_id, archive, 0),
            "uri": archive,
            "level": 0,
            "account_id": root.account_id,
            "vector": [0.1] * 4,
            "abstract": "backup parent",
        },
        ctx=root,
    )
    await fs.set_acl(
        archive,
        [owner, {"principal": "user:" + actor.user.user_id, "level": "manage"}],
        ctx=root,
        acl_mode=AclMode.RESTRICTED,
    )
    # These are actual ACL checks, never mocked permission results.
    action = AclAction.MANAGE if level == "manage" else AclAction.READ
    for uri in (target, staging):
        await fs._ensure_access(uri, actor, action=action)
    await fs._ensure_access(backup, actor, action=AclAction.WRITE)
    return actor


async def _restrict(case, actor, side, *, orphan=False, readable=False):
    fs, backend, root, target, staging, *_ = case
    base = target if side == "target" else staging
    hidden = base + ("/orphan-secret.md" if orphan else "/branch")
    owner = {"principal": "user:" + root.user.user_id, "level": "manage"}
    entries = [owner]
    if readable:
        entries.append({"principal": "user:" + actor.user.user_id, "level": "read"})
    if orphan:
        await backend.upsert(
            {
                "id": vector_record_id(root.account_id, hidden, 2),
                "uri": hidden,
                "level": 2,
                "account_id": root.account_id,
                "vector": [0.1] * 4,
                "abstract": "hidden orphan record",
                "search_tags": ["hidden=orphan-control"],
            },
            ctx=root,
        )
        await fs.acl_manager.set_acl(hidden, entries, root, acl_mode=AclMode.RESTRICTED)
        assert not await fs.exists(hidden, ctx=root)
    else:
        await fs.set_acl(hidden, entries, ctx=root, acl_mode=AclMode.RESTRICTED)
    if readable:
        await fs._ensure_access(hidden, actor, action=AclAction.READ)
    with pytest.raises(PermissionDeniedError):
        await fs._ensure_access(
            hidden, actor, action=AclAction.MANAGE if readable else AclAction.READ
        )
    return base, hidden


def _assert_generic(error, root, hidden):
    assert error.code == "PERMISSION_DENIED"
    exposed = json.dumps(
        {"message": str(error), "details": error.details}, sort_keys=True, default=str
    )
    assert "Tree contains inaccessible entries" in exposed
    assert root in exposed
    assert hidden not in exposed
    assert "current_revisions" not in exposed
    assert tree_replace.REVISION_PREFIX not in exposed
    for field in ("file_count", "directory_count", "total_bytes", "entry_count"):
        assert field not in exposed


async def _snapshot(case):
    fs, _, root, target, staging, backup, *_ = case
    trees = [await tree_replace.tree_revision(fs, uri, ctx=root) for uri in (target, staging)]
    records = await tree_replace._context_records(fs, "viking://resources", root)
    return {
        "trees": trees,
        "records": json.dumps(sorted(records, key=lambda r: r["id"]), sort_keys=True, default=str),
        "backup": await fs.exists(backup, ctx=root),
    }


@pytest.mark.parametrize("surface", ["function", "service"])
async def test_revision_denies_physical_descendant_without_reading_bytes(
    enabled_tree, monkeypatch, surface
):
    fs, backend, _, target, *_ = enabled_tree
    actor = await _actor(enabled_tree, level="read")
    _, hidden = await _restrict(enabled_tree, actor, "target")
    path = fs._uri_to_path(hidden, ctx=actor)
    original_cat, original_ls = fs._async_agfs.cat, fs._async_agfs.ls
    denied_reads = []

    async def cat(name, *args, **kwargs):
        if str(name) == path or str(name).startswith(path + "/"):
            denied_reads.append(("cat", str(name)))
        return await original_cat(name, *args, **kwargs)

    async def ls(name, *args, **kwargs):
        if str(name) == path or str(name).startswith(path + "/"):
            denied_reads.append(("ls", str(name)))
        return await original_ls(name, *args, **kwargs)

    monkeypatch.setattr(fs._async_agfs, "cat", cat)
    monkeypatch.setattr(fs._async_agfs, "ls", ls)
    with pytest.raises(PermissionDeniedError) as raised:
        if surface == "function":
            await tree_replace.tree_revision(fs, target, ctx=actor)
        else:
            await FSService(fs, backend).tree_revision(target, actor)
    _assert_generic(raised.value, target, hidden)
    assert not denied_reads


async def test_revision_denies_context_only_record_before_metadata_hash(enabled_tree, monkeypatch):
    fs, _, _, target, *_ = enabled_tree
    actor = await _actor(enabled_tree, level="read")
    _, hidden = await _restrict(enabled_tree, actor, "target", orphan=True)
    original = tree_replace._controls
    hidden_controls = []

    def controls(record):
        if record["uri"] == hidden:
            hidden_controls.append(True)
        return original(record)

    monkeypatch.setattr(tree_replace, "_controls", controls)
    with pytest.raises(PermissionDeniedError) as raised:
        await tree_replace.tree_revision(fs, target, ctx=actor)
    _assert_generic(raised.value, target, hidden)
    assert not hidden_controls


@pytest.mark.parametrize("side", ["target", "staging"])
@pytest.mark.parametrize(
    "orphan,readable", [(False, False), (False, True), (True, False), (True, True)]
)
async def test_replace_denial_precedes_stale_revision_and_has_no_effects(
    enabled_tree, side, orphan, readable
):
    fs, _, root, target, staging, backup, *_ = enabled_tree
    actor = await _actor(enabled_tree)
    denied_root, hidden = await _restrict(
        enabled_tree, actor, side, orphan=orphan, readable=readable
    )
    # Reprove the roots/backup permissions after the descendant policy changes.
    for uri in (target, staging):
        await fs._ensure_access(uri, actor, action=AclAction.MANAGE)
    await fs._ensure_access(backup, actor, action=AclAction.WRITE)
    before = await _snapshot(enabled_tree)
    expected_staging = (await tree_replace.tree_revision(fs, staging, ctx=root))["revision"]
    with pytest.raises(PermissionDeniedError) as raised:
        await tree_replace.replace_tree(
            fs,
            target_uri=target,
            staging_uri=staging,
            backup_uri=backup,
            expected_target_revision=tree_replace.REVISION_PREFIX + "0" * 64,
            expected_staging_revision=expected_staging,
            ctx=actor,
        )
    _assert_generic(raised.value, denied_root, hidden)
    assert await _snapshot(enabled_tree) == before


@pytest.mark.parametrize("as_root", [False, True])
async def test_authorized_revision_and_replace_preserve_existing_contract(enabled_tree, as_root):
    fs, _, root, target, staging, backup, old, _ = enabled_tree
    actor = await _actor(enabled_tree)
    ctx = root if as_root else actor
    current = await tree_replace.tree_revision(fs, target, ctx=root)
    assert await tree_replace.tree_revision(fs, target, ctx=ctx) == current
    staging_revision = (await tree_replace.tree_revision(fs, staging, ctx=ctx))["revision"]
    result = await tree_replace.replace_tree(
        fs,
        target_uri=target,
        staging_uri=staging,
        backup_uri=backup,
        expected_target_revision=current["revision"],
        expected_staging_revision=staging_revision,
        ctx=ctx,
    )
    assert result["content_committed"]
    assert result["previous_revision"] == current["revision"]
    assert result["staging_revision"] == staging_revision
    assert result["revision"] == (await tree_replace.tree_revision(fs, target, ctx=ctx))["revision"]
    assert not await fs.exists(staging, ctx=root)
    for name, data in old.items():
        assert await fs.read_file_bytes(backup + "/" + name, ctx=root) == data
