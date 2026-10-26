# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Account-bound resource imports under one native resources Tree lease.

Recompute is refused before mutation until the deferred embedding producer can
supply a persisted account-lifetime and source-revision commit fence.
"""

from openviking.server.error_mapping import is_not_found_error
from openviking.server.skill_package_update import clear_skill_package_contents
from openviking.service.task_tracker_concurrency import run_to_completion
from openviking.service.tree_replace import (
    _context_records,
    _finish_context_copy,
    _prepare_context_copy,
)
from openviking.storage.acl import AclAction
from openviking.storage.ovpack.format import join_uri
from openviking.storage.ovpack.manifest import manifest_entries_by_path
from openviking.storage.ovpack.vectors import choose_vector_restore_action, restore_vector_snapshot
from openviking.storage.pathlock_transfer import PathLockGuard
from openviking_cli.exceptions import ConflictError, FailedPreconditionError, InvalidArgumentError


class _SnapshotCollector:
    def __init__(self):
        self.records = []

    async def upsert(self, payload, *, ctx):
        self.records.append(payload)


async def import_resource_pack(
    fs,
    zf,
    *,
    root_uri,
    members,
    manifest,
    index_records,
    dense_vectors,
    conflict_action,
    vector_mode,
    vector_store,
    vector_config_resolver,
    ctx,
):
    """Import a validated resources archive; never clean up on failure."""
    if not callable(getattr(fs, "_acquire_resource_coverage", None)):
        raise FailedPreconditionError("Resource import requires qualified native account coverage")
    # Always resolve against this request's account, including ROOT callers.
    root_uri = fs._normalize_transfer_uri(root_uri).rstrip("/")
    root_path = fs._uri_to_path(root_uri, ctx=ctx)
    resources_path = fs._uri_to_path("viking://resources", ctx=ctx)
    if root_path != resources_path and not root_path.startswith(resources_path + "/"):
        raise InvalidArgumentError("Resource import cannot cross account roots")
    await fs._ensure_access(root_uri, ctx, action=AclAction.WRITE)
    admitted = await fs._capture_resource_identity(ctx)
    guard = PathLockGuard(fs._async_agfs)
    try:
        await guard.acquire(lambda: fs._acquire_resource_coverage(ctx, admitted))
        lease = guard.lease

        async def validate_owner():
            try:
                refreshed = await fs._async_agfs.pathlock_refresh(lease)
            except ValueError as exc:
                if not any(
                    message in str(exc)
                    for message in (
                        "owned lease capability does not match",
                        "unknown pathlock lease ref",
                    )
                ):
                    raise
                # Native ownership is authoritatively gone. Do not release a
                # capability we no longer own or mask this with a second refusal.
                guard.detach()
                raise FailedPreconditionError(
                    "Resource import lost its native owner lease"
                ) from exc
            if refreshed != "refreshed":
                raise FailedPreconditionError("Resource import lost its native owner lease")
            # Owner extension validates that the lease is still live and reruns
            # the authoritative account check without creating a target root.
            check = PathLockGuard(fs._async_agfs)
            try:
                await check.acquire(
                    lambda: fs._acquire_resource_coverage(
                        ctx,
                        admitted,
                        owner_lease_ref=lease,
                    )
                )
            finally:
                await check.close()

        await fs._ensure_access(root_uri, ctx, action=AclAction.WRITE)
        if root_uri != "viking://resources":
            await fs._ensure_transfer_parent_directory(root_path, root_uri, operation="import")
        try:
            stat = await fs._async_agfs.stat(root_path)
        except Exception as exc:
            if not is_not_found_error(exc):
                raise
            stat = None
        if stat is not None:
            if conflict_action == "skip":
                return root_uri
            if conflict_action == "fail":
                raise ConflictError(f"Resource already exists at {root_uri}", resource=root_uri)
            if not stat.get("isDir"):
                raise InvalidArgumentError("Resource pack target must be a directory")
            await fs._ensure_access(root_uri, ctx, action=AclAction.MANAGE)

        if vector_store is not None and vector_store is not fs._get_vector_store():
            raise FailedPreconditionError(
                "Resource import requires the filesystem's account context store"
            )
        action = await choose_vector_restore_action(
            manifest,
            index_records,
            dense_vectors,
            vector_store=vector_store,
            vector_config_resolver=vector_config_resolver,
            vector_mode=vector_mode,
            ctx=ctx,
        )
        if action != "restore":
            raise FailedPreconditionError(
                "Resource pack import requires a compatible dense vector snapshot; "
                "deferred recompute lacks an account-lifetime and source-revision commit fence",
                details={"uri": root_uri, "vector_mode": vector_mode, "mutation_started": False},
            )
        targets = [
            join_uri(root_uri, rel)
            for _, _, kind, rel in members
            if kind not in {"manifest", "internal"}
        ]
        await fs._ensure_access_many(targets, ctx, action=AclAction.WRITE)
        existing = await _context_records(fs, root_uri, ctx)
        await fs._ensure_access_many(
            [record["uri"] for record in existing],
            ctx,
            action=AclAction.MANAGE,
        )
        if stat is not None:
            # Raw traversal includes hidden descendants and fails on denied ACLs.
            uris = await fs._collect_uris(root_path, True, ctx=ctx, strict=True)
            await fs._ensure_access_many(uris, ctx, action=AclAction.MANAGE)

        collector = _SnapshotCollector()
        await restore_vector_snapshot(
            collector,
            root_uri,
            index_records,
            dense_vectors,
            manifest_entries_by_path(manifest),
            ctx,
        )
        # Reuse replacement's metadata/ACL retention rules, before any mutation.
        prepared = await _prepare_context_copy(
            fs,
            root_uri,
            root_uri,
            collector.records,
            ctx,
            retained=existing,
        )

        async def mutate():
            await validate_owner()
            if stat is not None:
                # Keep the root and its policy records until replacement is ready.
                await clear_skill_package_contents(
                    fs,
                    root_uri,
                    ctx=ctx,
                    lease_ref=lease,
                    preserve_index=True,
                )
            else:
                await fs._mkdir_transfer_root_exclusive(root_path, root_uri, ctx, lease)
            content = [member for member in members if member[2] not in {"manifest", "internal"}]
            content.sort(key=lambda member: (member[2] != "directory", member[3].count("/")))
            for _, zip_path, kind, relative in content:
                await validate_owner()
                uri = join_uri(root_uri, relative)
                if kind == "directory":
                    await fs.mkdir(uri, exist_ok=True, ctx=ctx, lease_ref=lease)
                else:
                    await fs.write_file_bytes(uri, zf.read(zip_path), ctx=ctx, lease_ref=lease)
            for record in prepared:
                await validate_owner()
                result = await vector_store.upsert(record, ctx=ctx)
                if not result:
                    raise RuntimeError("Resource import vector restore did not confirm persistence")
            await validate_owner()
            await _finish_context_copy(fs, root_uri, prepared, ctx)
            await validate_owner()

        # Settle in-flight native/index work before releasing on caller cancellation.
        # Failures retain partial contents and report failure, never foreign cleanup.
        await run_to_completion(mutate)
        return root_uri
    finally:
        await guard.close()
