# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Skill update transfers that never remove the directory holding the package lock."""

from typing import Any, Dict

from openviking.server.identity import RequestContext
from openviking.storage.internal_names import (
    MULTIWRITE_EXACT_LOCK_FILE_PREFIX,
    MULTIWRITE_INTERNAL_FILE_NAMES,
)
from openviking.storage.viking_fs import VikingFS


async def transfer_skill_package(
    viking_fs: VikingFS,
    source_uri: str,
    target_uri: str,
    *,
    ctx: RequestContext,
    lease_ref: Dict[str, Any],
    preserve_sidecar_bytes: bool = False,
    prepared_index_records: list[Dict[str, Any]] | None = None,
) -> None:
    """Copy files and transfer existing vectors, leaving both locked roots intact.

    The caller owns both tree locks and decides when the source can be cleared.
    Recovery copies may preserve sidecar bytes exactly; ordinary transfers keep
    the existing destination URI/link rewrite by default. A protected replacement
    can supply pre-materialized context records; those are copied before files,
    with source contexts retained. The default vector move path is unchanged.
    Do not use cp/mv here: their failure cleanup can delete the target root,
    which during rollback is the directory holding the live package lock.
    """
    source_uris = await viking_fs._prepare_transfer_entries(
        source_uri, target_uri, is_dir=True, move=True, ctx=ctx
    )
    if prepared_index_records is not None:
        # Every payload already has its final ACL. Publish it before file bytes,
        # retaining the source's contexts and avoiding a permissive move rewrite.
        store = viking_fs._get_vector_store()
        if prepared_index_records:
            if store is None:
                raise RuntimeError("protected transfer requires the context store")
            written = await store._upsert_many_raw(prepared_index_records, ctx=ctx)
            if len(written) != len(prepared_index_records):
                raise RuntimeError("protected transfer did not publish every context record")
    await viking_fs._copy_directory_under_tree_locks(
        viking_fs._uri_to_path(source_uri, ctx=ctx),
        viking_fs._uri_to_path(target_uri, ctx=ctx),
        old_uri=source_uri,
        new_uri=target_uri,
        ctx=ctx,
        lease_ref=lease_ref,
        **({"rewrite_semantic_sidecars": False} if preserve_sidecar_bytes else {}),
    )
    if prepared_index_records is not None:
        return
    # Move preserves the existing vector payloads and compensates on failure.
    # Keep the source files available even if index transfer/restoration fails.
    await viking_fs._update_vector_store_uris(
        source_uri, target_uri, recursive=True, ctx=ctx, source_uris=source_uris
    )


async def clear_skill_package_contents(
    viking_fs: VikingFS,
    root_uri: str,
    *,
    ctx: RequestContext,
    lease_ref: Dict[str, Any],
    preserve_index: bool = False,
) -> None:
    """Remove package entries and root vectors, keeping the root and its lock."""
    root_path = viking_fs._uri_to_path(root_uri, ctx=ctx)
    # Storage listing includes user-hidden files that the public Skill listing
    # omits. Runtime control entries must remain owned by the storage layer.
    entries = await viking_fs._async_agfs.ls(
        root_path, fs_ctx=viking_fs._pathlock_fs_ctx(ctx, lease_ref)
    )
    for entry in entries:
        name = entry.get("name", "")
        if (
            not name
            or name in {".", ".."}
            or name in MULTIWRITE_INTERNAL_FILE_NAMES
            or name.startswith(MULTIWRITE_EXACT_LOCK_FILE_PREFIX)
        ):
            continue
        await viking_fs.rm(
            f"{root_uri}/{name}",
            recursive=bool(entry.get("isDir", False)),
            ctx=ctx,
            lease_ref=lease_ref,
            **({"no_vector": True} if preserve_index else {}),
        )
    if not preserve_index:
        await viking_fs._delete_from_vector_store([root_uri], ctx=ctx)
