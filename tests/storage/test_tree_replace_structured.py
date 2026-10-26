# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Structured sidecar replacement using real local storage and enabled ACLs.

These cover destination byte manifests, not ancestor lifetime or metadata
preservation. ACLs are enabled to exercise the actual indexed transfer path.
"""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from openviking.service import tree_replace
from openviking.storage.abstract_overview import (
    parse_abstract_overview,
    render_abstract_overview,
    rewrite_abstract_overview_for_transfer,
)
from openviking.storage.acl import AclManager
from openviking.storage.vector_ids import vector_record_id
from openviking_cli.exceptions import OpenVikingError
from tests.storage.test_transfer_merge_binding import (
    _EnabledAclConfig,
    binding_fs,
    indexed_fs,
    root_ctx,
)


@pytest_asyncio.fixture
async def enabled_tree(indexed_fs):
    fs, backend = indexed_fs
    ctx = root_ctx()
    fs.acl_manager = backend.acl_manager = AclManager(backend, _EnabledAclConfig())
    assert await fs.acl_manager.is_enabled(ctx.account_id)
    target = "viking://resources/published/doc"
    staging = "viking://resources/prepared/input"
    backup = "viking://resources/archive/previous"
    for parent in ("published", "prepared", "archive"):
        await fs.mkdir(f"viking://resources/{parent}", ctx=ctx, exist_ok=True)

    async def seed(base, label):
        content = {
            "one.md": f"{label} one".encode(),
            "branch/two.md": f"{label} two".encode(),
            "branch/deeper/three.md": f"{label} three".encode(),
        }
        for relative, data in list(content.items()):
            await fs.write_file_bytes(f"{base}/{relative}", data, ctx=ctx)
        for directory in ("", "branch", "branch/deeper"):
            uri = f"{base}/{directory}" if directory else base
            body = f"[self]({uri}) [one]({base}/one.md) [two]({base}/branch/two.md)\n"
            for name, level in ((".abstract.md", 0), (".overview.md", 1)):
                data = render_abstract_overview(
                    level, uri, body,
                    {"source": {"kind": "file", "uri": "file:///fixture/input.md"}},
                ).encode()
                # Recovery must retain even noncanonical, valid YAML bytes.
                if label == "old":
                    data = data.replace(b"---\n", b"---\r\n", 1)
                relative = f"{directory}/{name}" if directory else name
                content[relative] = data
                path = fs._uri_to_path(f"{uri}/{name}", ctx=ctx)
                await fs._async_agfs.write(path, data)
                await backend.upsert(
                    {
                        "id": vector_record_id(ctx.account_id, uri, level),
                        "uri": uri, "level": level, "account_id": ctx.account_id,
                        "vector": [0.1] * 4, "content": data.decode(),
                        "abstract": body,
                    }, ctx=ctx,
                )
        return content

    old = await seed(target, "old")
    incoming = await seed(staging, "new")
    yield fs, backend, ctx, target, staging, backup, old, incoming


async def revision(fs, uri, ctx):
    return (await tree_replace.tree_revision(fs, uri, ctx=ctx))["revision"]


async def replace_case(case, **kwargs):
    fs, _, ctx, target, staging, backup, _, _ = case
    return await tree_replace.replace_tree(
        fs, target_uri=target, staging_uri=staging, backup_uri=backup,
        expected_target_revision=await revision(fs, target, ctx),
        expected_staging_revision=await revision(fs, staging, ctx),
        ctx=ctx, **kwargs,
    )


async def raw(fs, uri, ctx):
    return await fs._async_agfs.cat(fs._uri_to_path(uri, ctx=ctx))


async def test_structured_install_predicts_destination_and_preserves_raw_backup(enabled_tree):
    fs, _, ctx, target, staging, backup, old, incoming = enabled_tree
    old_revision = await revision(fs, target, ctx)
    incoming_revision = await revision(fs, staging, ctx)
    result = await replace_case(enabled_tree)
    assert result["content_committed"]
    assert result["previous_revision"] == old_revision
    assert result["staging_revision"] == incoming_revision
    assert result["revision"] == await revision(fs, target, ctx)
    assert result["revision"] != incoming_revision
    # Backup ACL anchors the original effective policy at its new parent;
    # content bytes below remain exact, while the full metadata revision differs.
    assert await revision(fs, backup, ctx) != old_revision
    for relative, original in old.items():
        assert await raw(fs, f"{backup}/{relative}", ctx) == original
    for relative, original in incoming.items():
        name = relative.rsplit("/", 1)[-1]
        expected = original
        if name in (".abstract.md", ".overview.md"):
            parent = relative.rpartition("/")[0]
            source_dir = f"{staging}/{parent}" if parent else staging
            target_dir = f"{target}/{parent}" if parent else target
            expected = rewrite_abstract_overview_for_transfer(
                original, level=0 if name == ".abstract.md" else 1,
                source_dir_uri=source_dir, target_dir_uri=target_dir,
                source_scope_uri=staging, target_scope_uri=target,
            ).encode()
            document = parse_abstract_overview(expected)
            assert document.metadata["directory"] == target_dir + "/"
            assert staging not in document.body
            assert target in document.body
        assert await raw(fs, f"{target}/{relative}", ctx) == expected


async def test_structured_rollback_restores_exact_raw_bytes(enabled_tree, monkeypatch):
    fs, _, ctx, target, staging, backup, old, incoming = enabled_tree
    old_revision = await revision(fs, target, ctx)
    original = tree_replace.transfer_skill_package

    async def install_then_fail(fs, source, dest, **kwargs):
        await original(fs, source, dest, **kwargs)
        if source == staging:
            raise OSError("injected failure after structured install")

    monkeypatch.setattr(tree_replace, "transfer_skill_package", install_then_fail)
    with pytest.raises(OpenVikingError) as raised:
        await replace_case(enabled_tree)
    assert raised.value.details["state"] == "rolled_back"
    assert not raised.value.details["content_committed"]
    assert await revision(fs, target, ctx) == old_revision
    for relative, data in old.items():
        assert await raw(fs, f"{target}/{relative}", ctx) == data
        assert await raw(fs, f"{backup}/{relative}", ctx) == data
    for relative, data in incoming.items():
        assert await raw(fs, f"{staging}/{relative}", ctx) == data


async def test_structured_corrupt_install_is_rejected_and_rolled_back(enabled_tree, monkeypatch):
    fs, _, ctx, target, staging, _, old, _ = enabled_tree
    original = tree_replace.transfer_skill_package

    async def corrupt_install(fs, source, dest, **kwargs):
        await original(fs, source, dest, **kwargs)
        if source == staging:
            await fs._async_agfs.write(
                fs._uri_to_path(f"{target}/branch/.overview.md", ctx=ctx),
                b"corrupted destination",
                fs_ctx=fs._pathlock_fs_ctx(ctx, kwargs["lease_ref"]),
            )

    monkeypatch.setattr(tree_replace, "transfer_skill_package", corrupt_install)
    with pytest.raises(OpenVikingError) as raised:
        await replace_case(enabled_tree)
    assert raised.value.details["state"] == "rolled_back"
    assert not raised.value.details["content_committed"]
    for relative, data in old.items():
        assert await raw(fs, f"{target}/{relative}", ctx) == data


async def test_structured_prediction_refuses_changed_raw_input(enabled_tree):
    fs, _, ctx, target, staging, _, _, _ = enabled_tree
    path = fs._uri_to_path(staging, ctx=ctx)
    entries = await tree_replace._tree_entries(fs, path, None)
    await fs._async_agfs.write(f"{path}/.overview.md", b"changed after raw revision")
    with pytest.raises(OpenVikingError) as raised:
        await tree_replace._expected_transfer_entries(
            fs, staging, target, entries, None, source_path=path
        )
    assert raised.value.code == "CONFLICT"


async def test_structured_prediction_fails_closed_on_unknown_read(enabled_tree, monkeypatch):
    fs, _, ctx, target, staging, _, old, _ = enabled_tree
    path = fs._uri_to_path(staging, ctx=ctx)
    entries = await tree_replace._tree_entries(fs, path, None)
    assert entries is not None
    monkeypatch.setattr(fs._async_agfs, "cat", AsyncMock(side_effect=OSError("unreadable")))
    with pytest.raises(OSError, match="unreadable"):
        await tree_replace._expected_transfer_entries(
            fs, staging, target, entries, None, source_path=path
        )
