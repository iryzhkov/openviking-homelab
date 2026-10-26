"""Prepare resource move policy before exposing any destination bytes."""

import hashlib
from dataclasses import replace

from openviking.core.context import ContextLevel
from openviking.service.account_identity import recheck_account_identity
from openviking.storage.abstract_overview import (
    ABSTRACT_OVERVIEW_FILENAMES,
    rewrite_abstract_overview_for_transfer,
)
from openviking.storage.acl import AclAction, AclMode
from openviking.storage.vector_migration import rewrite_vector_record
from openviking.utils.time_utils import get_current_timestamp
from openviking_cli.exceptions import ConflictError, FailedPreconditionError, OpenVikingError


async def move_prepared_resource(
    fs,
    *,
    source,
    target,
    ctx,
    admitted,
    lease,
    is_dir,
    physical_uris,
    vector_mode,
    copy_semantic_sidecars,
    operation_id,
):
    """Keep source policy through removal; retain protected copies on failure."""
    ctx = fs._ctx_or_default(ctx)
    if not (fs._is_shared_resource_uri(source) and fs._is_shared_resource_uri(target)):
        raise FailedPreconditionError("ACL-protected moves require resource source and destination")
    if vector_mode != "transfer" or not copy_semantic_sidecars:
        raise FailedPreconditionError("ACL-protected moves require a complete indexed transfer")
    store = fs._get_vector_store()
    manager = fs.acl_manager
    if store is None or manager is None:
        raise FailedPreconditionError("ACL-protected moves require indexed source policy")
    source, target = source.rstrip("/"), target.rstrip("/")
    fs_ctx = fs._pathlock_fs_ctx(ctx, lease)

    async def check():
        if await fs._async_agfs.pathlock_refresh(lease) != "refreshed":
            raise ConflictError("Resource move lost original ownership")
        await recheck_account_identity(fs._account_identity_manager, ctx.account_id, admitted)

    async def scan(uri, recursive):
        records, _ = await store._scan_uri_transfer_scope(
            replace(ctx, bypass_acl=True), uri, recursive=recursive, include_full_records=True
        )
        return records

    await check()
    records = await scan(source, is_dir)
    if not any(record["uri"] == source for record in records):
        raise FailedPreconditionError("ACL-protected move source has no indexed policy root")
    await fs._ensure_access_many(
        list(dict.fromkeys([*physical_uris, *(r["uri"] for r in records)])),
        ctx,
        action=AclAction.MANAGE if is_dir else AclAction.WRITE,
    )
    if await scan(target, True):
        raise ConflictError("Move destination index scope is occupied", resource=target)
    anchor = await manager.resolve(source, ctx)
    payloads = [
        rewrite_vector_record(
            record,
            source_uri=source,
            target_uri=target,
            ctx=ctx,
            mode="move",
            timestamp=get_current_timestamp(),
        )
        for record in records
    ]
    _, payloads = await manager.prepare_subtree_records(
        target, payloads, ctx, root_direct=anchor.permissions, root_mode=AclMode.RESTRICTED
    )
    bound = await store._get_backend_for_context(ctx)
    payloads = await bound._async_adapter.run(bound._prepare_upsert_payloads, payloads)
    # Read every source file before effects and predict existing sidecar rewriting.
    entries = []
    for uri in physical_uris:
        await check()
        path = fs._uri_to_path(uri, ctx=ctx)
        stat = await fs._async_agfs.stat(path)
        directory = bool(stat.get("isDir", False))
        relative = uri.rstrip("/")[len(source) :]
        expected = None
        if not directory:
            raw = await fs._async_agfs.cat(path, fs_ctx=fs_ctx)
            data = (
                raw
                if isinstance(raw, bytes)
                else bytes(raw)
                if isinstance(raw, bytearray)
                else str(raw).encode()
            )
            name = uri.rsplit("/", 1)[-1]
            if is_dir and name in ABSTRACT_OVERVIEW_FILENAMES:
                data = rewrite_abstract_overview_for_transfer(
                    data,
                    level=ContextLevel.ABSTRACT
                    if name == ".abstract.md"
                    else ContextLevel.OVERVIEW,
                    source_dir_uri=uri.rsplit("/", 1)[0],
                    target_dir_uri=(target + relative).rsplit("/", 1)[0],
                    source_scope_uri=source,
                    target_scope_uri=target,
                ).encode()
            expected = data
        digest = hashlib.sha256(expected).digest() if expected is not None else None
        rewritten = (
            expected if is_dir and uri.rsplit("/", 1)[-1] in ABSTRACT_OVERVIEW_FILENAMES else None
        )
        entries.append((uri, target + relative, directory, digest, rewritten))

    phase = "publish_policy"
    destination_verified = False
    source_removed = False
    try:
        await check()
        # Recheck both absences immediately before publishing any owned record.
        await fs._ensure_transfer_destination_absent(fs._uri_to_path(target, ctx=ctx), target)
        if await scan(target, True):
            raise ConflictError("Move destination index scope is occupied", resource=target)
        for offset in range(0, len(payloads), 100):
            await check()
            batch = payloads[offset : offset + 100]
            written = await store._upsert_many_raw(batch, ctx=ctx)
            if len(written) != len(batch):
                raise RuntimeError("Move did not publish complete destination policy")
        actual = await store._strict_transfer_get(ctx, [r["id"] for r in payloads])
        actual = await bound._async_adapter.run(bound._prepare_upsert_payloads, actual)

        def stored_fields(record):
            # Native string defaults read back as empty strings, even when the
            # write payload omitted them. All nonempty and ACL fields remain exact.
            return {
                key: value for key, value in record.items() if value is not None and value != ""
            }

        by_id = {r["id"]: stored_fields(r) for r in actual}
        if any(by_id.get(r["id"]) != stored_fields(r) for r in payloads):
            raise RuntimeError("Move destination policy verification failed")
        phase = "copy_content"
        for old, new, directory, _, rewritten in entries:
            await check()
            new_path = fs._uri_to_path(new, ctx=ctx)
            if directory:
                if new == target:
                    await fs._mkdir_transfer_root_exclusive(new_path, new, ctx, lease)
                else:
                    await fs._async_agfs.mkdir(new_path, fs_ctx=fs_ctx)
            elif is_dir and old.rsplit("/", 1)[-1] in ABSTRACT_OVERVIEW_FILENAMES:
                await fs._async_agfs.write(new_path, rewritten, fs_ctx=fs_ctx)
            else:
                await fs._async_agfs.cp(
                    fs._uri_to_path(old, ctx=ctx), new_path, recursive=False, fs_ctx=fs_ctx
                )
        phase = "verify_content"
        for _, new, directory, expected_digest, _ in entries:
            await check()
            path = fs._uri_to_path(new, ctx=ctx)
            if directory:
                if not (await fs._async_agfs.stat(path)).get("isDir", False):
                    raise RuntimeError("Move destination directory verification failed")
            else:
                raw = await fs._async_agfs.cat(path, fs_ctx=fs_ctx)
                data = (
                    raw
                    if isinstance(raw, bytes)
                    else bytes(raw)
                    if isinstance(raw, bytearray)
                    else str(raw).encode()
                )
                if hashlib.sha256(data).digest() != expected_digest:
                    raise RuntimeError("Move destination content verification failed")
        destination_verified = True
        phase = "remove_source_content"
        await check()
        source_removed = None  # Native removal failure may have made partial progress.
        await fs._async_agfs.rm(fs._uri_to_path(source, ctx=ctx), recursive=is_dir, fs_ctx=fs_ctx)
        # Successful removal alone is insufficient evidence for policy deletion.
        await fs._ensure_transfer_destination_absent(fs._uri_to_path(source, ctx=ctx), source)
        source_removed = True
        phase = "remove_source_policy"
        ids = [r["id"] for r in records]
        for offset in range(0, len(ids), 100):
            await check()
            batch = ids[offset : offset + 100]
            await store._strict_transfer_delete(ctx, batch)
            if await store._strict_transfer_get(ctx, batch):
                raise RuntimeError("Move retained source policy records")
    except BaseException as exc:
        # Never delete potentially foreign bytes or drop policy on a failed observation.
        # Cancellation has the same truthful partial-state contract as storage faults.
        raise OpenVikingError(
            "Protected move did not complete; restrictive policy was retained",
            code="MOVE_INCOMPLETE",
            details={
                "operation_id": operation_id,
                "phase": phase,
                "content_committed": destination_verified,
                "source_removed": source_removed,
                "source_uri": source,
                "target_uri": target,
                "retained_policy_uris": [source, target],
                "cause": type(exc).__name__,
            },
        ) from exc
    return {
        "operation_id": operation_id,
        "operation": "move",
        "from": source,
        "to": target,
        "recursive": is_dir,
        "phase": "completed",
        "files_created": len(entries),
        "files_deleted": len(entries),
        "vectors": {
            "scanned": len(records),
            "written": len(payloads),
            "deleted": len(records),
            "restored": 0,
            "batches": (len(records) + 99) // 100,
        },
    }
