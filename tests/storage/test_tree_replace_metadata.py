"""Real local-store regressions for replacement metadata and ACL policy."""

import pytest

from openviking.service import tree_replace
from openviking.storage.acl import AclMode
from openviking.storage.content_write import ContentWriteCoordinator
from openviking.storage.errors import LockAcquisitionError, ResourceBusyError
from openviking.server.identity import RequestContext, Role
from openviking_cli.exceptions import OpenVikingError, PermissionDeniedError
from openviking_cli.session.user_id import UserIdentifier
from tests.storage.test_tree_replace_structured import (
    binding_fs, indexed_fs, enabled_tree, revision, raw, replace_case,
)


async def test_tags_must_participate_in_held_tree_lease(enabled_tree):
    fs, _, ctx, target, _, _, _, _ = enabled_tree
    lease = await fs._async_agfs.pathlock_acquire_tree(fs._uri_to_path(target, ctx=ctx))
    try:
        with pytest.raises((LockAcquisitionError, ResourceBusyError)):
            await ContentWriteCoordinator(fs).set_tags(
                uri=target, tags=["owner=concurrent"], mode="replace", recursive=False, ctx=ctx
            )
    finally:
        await fs._async_agfs.pathlock_release(lease)


@pytest.mark.parametrize("mutation", ["tags", "acl"])
async def test_raw_revision_must_cover_user_controlled_metadata(enabled_tree, mutation):
    fs, backend, ctx, target, _, _, _, _ = enabled_tree
    before = await revision(fs, target, ctx)
    if mutation == "tags":
        changed = await backend.update_search_tags(
            target, ["owner=changed"], mode="replace", levels=[0, 1], ctx=ctx
        )
        assert changed
    else:
        await fs.set_acl(
            target, [{"principal": f"user:{ctx.user.user_id}", "level": "manage"}],
            ctx=ctx, acl_mode=AclMode.RESTRICTED,
        )
    assert await revision(fs, target, ctx) != before


async def test_incoming_defaults_must_not_loosen_target_policy(enabled_tree):
    fs, _, ctx, target, _, _, _, _ = enabled_tree
    outsider = RequestContext(
        user=UserIdentifier(ctx.account_id, "outsider"), role=Role(Role.USER)
    )
    await fs.set_acl(
        target, [{"principal": f"user:{ctx.user.user_id}", "level": "manage"}],
        ctx=ctx, acl_mode=AclMode.RESTRICTED,
    )
    with pytest.raises(PermissionDeniedError):
        await fs.read_file_bytes(f"{target}/one.md", ctx=outsider)
    await replace_case(enabled_tree)
    with pytest.raises(PermissionDeniedError):
        await fs.read_file_bytes(f"{target}/one.md", ctx=outsider)


async def test_backup_in_public_parent_must_keep_original_effective_acl(enabled_tree):
    fs, backend, ctx, target, _, backup, _, _ = enabled_tree
    parent = "viking://resources/published"
    await backend.upsert({
        "id": parent, "uri": parent, "level": 0, "vector": [0.1] * 4,
        "account_id": ctx.account_id, "abstract": "parent",
    }, ctx=ctx)
    await fs.set_acl(
        parent, [{"principal": f"user:{ctx.user.user_id}", "level": "manage"}],
        ctx=ctx, acl_mode=AclMode.RESTRICTED,
    )
    outsider = RequestContext(
        user=UserIdentifier(ctx.account_id, "outsider"), role=Role(Role.USER)
    )
    with pytest.raises(PermissionDeniedError):
        await fs.read_file_bytes(f"{target}/one.md", ctx=outsider)
    await replace_case(enabled_tree)
    with pytest.raises(PermissionDeniedError):
        await fs.read_file_bytes(f"{backup}/one.md", ctx=outsider)


async def test_acl_refresh_failure_must_prevent_usable_success(enabled_tree, monkeypatch):
    fs, _, ctx, target, _, _, _, _ = enabled_tree

    async def fail_refresh(*args, **kwargs):
        raise OSError("injected inherited ACL refresh failure")

    monkeypatch.setattr(fs.acl_manager, "refresh_context_subtree", fail_refresh)
    with pytest.raises(OpenVikingError):
        await replace_case(enabled_tree)


async def _restrict_mixed(case):
    fs, backend, ctx, target, staging, _, _, _ = case
    owner = {"principal": f"user:{ctx.user.user_id}", "level": "manage"}
    reader = {"principal": "user:reader", "level": "read"}
    await fs.set_acl(target, [owner, reader], ctx=ctx, acl_mode=AclMode.RESTRICTED)
    await fs.set_acl(f"{target}/branch", [owner], ctx=ctx, acl_mode=AclMode.RESTRICTED)
    removed = f"{target}/removed/secret.md"
    await fs.write_file_bytes(removed, b"removed private bytes", ctx=ctx)
    await backend.upsert({
        "id": f"{target}/removed", "uri": f"{target}/removed", "level": 0,
        "account_id": ctx.account_id, "vector": [0.1] * 4, "abstract": "private",
    }, ctx=ctx)
    await fs.set_acl(f"{target}/removed", [owner], ctx=ctx, acl_mode=AclMode.RESTRICTED)
    await ContentWriteCoordinator(fs).set_tags(
        uri=target, tags=["owner=target"], mode="replace", recursive=False, ctx=ctx
    )
    await ContentWriteCoordinator(fs).set_tags(
        uri=staging, tags=["owner=staging"], mode="replace", recursive=False, ctx=ctx
    )
    return RequestContext(user=UserIdentifier(ctx.account_id, "reader"), role=Role(Role.USER))


async def test_retained_tags_mixed_policies_and_removed_backup_descendants(enabled_tree):
    fs, backend, ctx, target, _, backup, _, _ = enabled_tree
    reader = await _restrict_mixed(enabled_tree)
    await replace_case(enabled_tree)
    assert await fs.read_file_bytes(f"{target}/one.md", ctx=reader) == b"new one"
    for uri in (f"{target}/branch/two.md", f"{backup}/removed/secret.md"):
        with pytest.raises(PermissionDeniedError):
            await fs.read_file_bytes(uri, ctx=reader)
    assert await fs.read_file_bytes(f"{backup}/one.md", ctx=reader) == b"old one"
    assert not await fs.exists(f"{target}/removed", ctx=ctx)
    assert not await backend.get_context_by_uri(f"{target}/removed", ctx=ctx)
    records = await tree_replace._context_records(fs, target, ctx)
    root_records = [record for record in records if record["uri"] == target]
    assert root_records and all(record["search_tags"] == ["owner=target"] for record in root_records)
    effective = await fs.acl_manager.resolve(f"{target}/branch/deeper", ctx)
    assert effective.permissions.principals_for(tree_replace.AclAction.READ) == {f"user:{ctx.user.user_id}"}


async def test_backup_policy_holds_during_copy_before_post_refresh(enabled_tree, monkeypatch):
    fs, _, ctx, _, _, backup, _, _ = enabled_tree
    reader = await _restrict_mixed(enabled_tree)
    outsider = RequestContext(user=UserIdentifier(ctx.account_id, "outsider"), role=Role(Role.USER))
    original = fs._copy_directory_under_tree_locks
    checked = []

    async def inspect_copy(*args, **kwargs):
        result = await original(*args, **kwargs)
        if kwargs.get("new_uri") == backup:
            with pytest.raises(PermissionDeniedError):
                await fs.read_file_bytes(f"{backup}/one.md", ctx=outsider)
            with pytest.raises(PermissionDeniedError):
                await fs.read_file_bytes(f"{backup}/removed/secret.md", ctx=reader)
            checked.append(True)
        return result

    monkeypatch.setattr(fs, "_copy_directory_under_tree_locks", inspect_copy)
    await replace_case(enabled_tree)
    assert checked == [True]


async def test_rollback_restores_policy_and_tags_before_reporting_rolled_back(enabled_tree, monkeypatch):
    fs, _, ctx, target, staging, backup, _, _ = enabled_tree
    reader = await _restrict_mixed(enabled_tree)
    before = await revision(fs, target, ctx)
    original = tree_replace.transfer_skill_package

    async def install_fail(fs, source, dest, **kwargs):
        await original(fs, source, dest, **kwargs)
        if source == staging:
            raise OSError("after protected install")

    monkeypatch.setattr(tree_replace, "transfer_skill_package", install_fail)
    with pytest.raises(OpenVikingError) as raised:
        await replace_case(enabled_tree)
    assert raised.value.details["state"] == "rolled_back"
    assert await revision(fs, target, ctx) == before
    assert await fs.read_file_bytes(f"{target}/one.md", ctx=reader) == b"old one"
    for uri in (f"{target}/branch/two.md", f"{backup}/removed/secret.md"):
        with pytest.raises(PermissionDeniedError):
            await fs.read_file_bytes(uri, ctx=reader)


async def test_rollback_acl_refresh_failure_retains_private_copies(enabled_tree, monkeypatch):
    fs, _, ctx, target, staging, backup, _, _ = enabled_tree
    await _restrict_mixed(enabled_tree)
    original_transfer = tree_replace.transfer_skill_package
    original_refresh = fs.acl_manager.refresh_context_subtree

    async def install_fail(fs, source, dest, **kwargs):
        await original_transfer(fs, source, dest, **kwargs)
        if source == staging:
            raise OSError("trigger rollback")

    async def refresh_fail_on_target(uri, ctx):
        if uri == target:
            raise OSError("rollback refresh unavailable")
        return await original_refresh(uri, ctx)

    monkeypatch.setattr(tree_replace, "transfer_skill_package", install_fail)
    monkeypatch.setattr(fs.acl_manager, "refresh_context_subtree", refresh_fail_on_target)
    with pytest.raises(OpenVikingError) as raised:
        await replace_case(enabled_tree)
    details = raised.value.details
    assert details["state"] == "rollback_failed"
    assert not details["content_committed"]
    assert {item["uri"] for item in details["retained"]} >= {backup, staging}
    outsider = RequestContext(user=UserIdentifier(ctx.account_id, "outsider"), role=Role(Role.USER))
    for uri in (f"{target}/one.md", f"{backup}/one.md"):
        with pytest.raises(PermissionDeniedError):
            await fs.read_file_bytes(uri, ctx=outsider)


@pytest.mark.parametrize("directory", [False, True])
async def test_tag_mutation_holds_lease_through_updates_and_accepts_outer_coverage(
    enabled_tree, monkeypatch, directory
):
    fs, backend, ctx, target, _, _, _, _ = enabled_tree
    uri = target if directory else f"{target}/one.md"
    if not directory:
        await backend.upsert({
            "id": uri, "uri": uri, "level": 2, "vector": [0.1] * 4,
            "account_id": ctx.account_id, "abstract": "file",
        }, ctx=ctx)
    path = fs._uri_to_path(uri, ctx=ctx)
    original = backend.update_search_tags
    updates = []

    async def update_under_lease(*args, **kwargs):
        with pytest.raises(LockAcquisitionError):
            await fs._async_agfs.pathlock_acquire_tree(path)
        updates.append(True)
        return await original(*args, **kwargs)

    monkeypatch.setattr(backend, "update_search_tags", update_under_lease)
    lease = await fs._async_agfs.pathlock_acquire_tree(path)
    try:
        await ContentWriteCoordinator(fs).set_tags(
            uri=uri, tags=["lease=owned"], ctx=ctx, lease_ref=lease
        )
    finally:
        await fs._async_agfs.pathlock_release(lease)
    assert updates
    probe = await fs._async_agfs.pathlock_acquire_tree(path)
    await fs._async_agfs.pathlock_release(probe)


async def test_tag_mutation_reauthorizes_after_acquisition(enabled_tree, monkeypatch):
    fs, _, ctx, target, _, _, _, _ = enabled_tree
    actor = RequestContext(user=UserIdentifier(ctx.account_id, "writer"), role=Role(Role.USER))
    original = fs._async_agfs.pathlock_acquire_batch
    changed = False

    async def acquire_after_acl_change(*args, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            await fs.set_acl(
                target, [{"principal": f"user:{ctx.user.user_id}", "level": "manage"}],
                ctx=ctx, acl_mode=AclMode.RESTRICTED,
            )
        return await original(*args, **kwargs)

    monkeypatch.setattr(fs._async_agfs, "pathlock_acquire_batch", acquire_after_acl_change)
    with pytest.raises(PermissionDeniedError):
        await ContentWriteCoordinator(fs).set_tags(uri=target, tags=["owner=writer"], ctx=actor)
    records = await tree_replace._context_records(fs, target, ctx)
    assert all("owner=writer" not in (record.get("search_tags") or []) for record in records)


async def test_revision_excludes_embeddings_and_volatile_index_fields(enabled_tree):
    fs, backend, ctx, target, _, _, _, _ = enabled_tree
    before = await revision(fs, target, ctx)
    records = await tree_replace._context_records(fs, target, ctx)
    for record in records:
        record["vector"] = [0.8] * 4
        record["updated_at"] = "2026-10-26T00:00:00Z"
        record["active_count"] = 123
    await backend._upsert_many_raw(records, ctx=ctx)
    assert await revision(fs, target, ctx) == before


async def test_unindexed_retained_policy_is_refused_without_content_changes(enabled_tree):
    from openviking_cli.exceptions import InvalidArgumentError
    fs, backend, ctx, target, staging, _, _, _ = enabled_tree
    for uri, data in ((f"{target}/policy.md", b"old private"), (f"{staging}/policy.md", b"new")):
        await fs.write_file_bytes(uri, data, ctx=ctx)
    uri = f"{target}/policy.md"
    await backend.upsert({
        "id": uri, "uri": uri, "level": 2, "vector": [0.1] * 4,
        "account_id": ctx.account_id, "abstract": "policy",
    }, ctx=ctx)
    await fs.set_acl(
        uri, [{"principal": f"user:{ctx.user.user_id}", "level": "manage"}],
        ctx=ctx, acl_mode=AclMode.RESTRICTED,
    )
    with pytest.raises(InvalidArgumentError, match="indexed retained metadata"):
        await replace_case(enabled_tree)
    assert await fs.read_file_bytes(uri, ctx=ctx) == b"old private"


async def test_unknown_metadata_snapshot_refuses_before_content_changes(enabled_tree, monkeypatch):
    fs, backend, ctx, target, staging, backup, _, _ = enabled_tree
    await _restrict_mixed(enabled_tree)
    target_revision = await revision(fs, target, ctx)
    staging_revision = await revision(fs, staging, ctx)

    async def unreadable(*args, **kwargs):
        raise OSError("metadata snapshot unavailable")

    monkeypatch.setattr(backend, "_scan_uri_transfer_scope", unreadable)
    with pytest.raises(OSError, match="metadata snapshot unavailable"):
        await tree_replace.replace_tree(
            fs, target_uri=target, staging_uri=staging, backup_uri=backup,
            expected_target_revision=target_revision,
            expected_staging_revision=staging_revision, ctx=ctx,
        )
    assert await fs.read_file_bytes(f"{target}/one.md", ctx=ctx) == b"old one"
    outsider = RequestContext(user=UserIdentifier(ctx.account_id, "outsider"), role=Role(Role.USER))
    with pytest.raises(PermissionDeniedError):
        await fs.read_file_bytes(f"{target}/one.md", ctx=outsider)


@pytest.mark.parametrize("failed_copy", ["staging", "backup"])
async def test_cleanup_failure_cannot_remove_private_copy_policy(
    enabled_tree, monkeypatch, failed_copy
):
    fs, _, ctx, target, staging, backup, _, _ = enabled_tree
    await _restrict_mixed(enabled_tree)
    await fs.set_acl(
        staging, [{"principal": f"user:{ctx.user.user_id}", "level": "manage"}],
        ctx=ctx, acl_mode=AclMode.RESTRICTED,
    )
    retained = staging if failed_copy == "staging" else backup
    path = fs._uri_to_path(retained, ctx=ctx)
    original = fs._async_agfs.rm

    async def refuse_cleanup(candidate, *args, **kwargs):
        if candidate == path:
            raise OSError("private copy cleanup unavailable")
        return await original(candidate, *args, **kwargs)

    monkeypatch.setattr(fs._async_agfs, "rm", refuse_cleanup)
    result = await replace_case(enabled_tree, retain_backup=False)
    assert result["content_committed"]
    assert result["cleanup"]["status"] == "failed"
    assert retained in result["cleanup"]["retained"]
    outsider = RequestContext(user=UserIdentifier(ctx.account_id, "outsider"), role=Role(Role.USER))
    for uri in (f"{retained}/one.md", f"{target}/one.md"):
        with pytest.raises(PermissionDeniedError):
            await fs.read_file_bytes(uri, ctx=outsider)
