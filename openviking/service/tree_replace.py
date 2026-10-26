# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Guarded replacement of one resource directory by a prepared staging directory.

The replacement reuses the storage primitives of the Skill update flow: one Tree
lease over the target, staging and backup directories, file copies that keep
every locked root in place, and vector URI transfer instead of re-embedding.

Guarantee, stated exactly:

* Preconditions (tree revisions of target and staging, absent backup) are
  checked under the lease, before the first change. A mismatch returns 409
  and changes nothing except removing lock placeholders this call created.
* While the lease is held, no backend writer that takes RAGFS path locks can
  change any of the three trees. Writers that bypass path locks, including
  direct disk writers, are not covered.
* The steps are not crash-atomic and not invisible to readers: a reader can see
  the target empty or partially installed while the lease is held. The old
  tree is copied to the backup and verified by revision before the target is
  cleared. Staging is never cleared before the installed target matches the
  predicted destination manifest, including rewritten structured sidecars.
  The input revisions always describe the original raw bytes.
* On a failure the old tree is restored from the verified backup when
  possible. Copies are never deleted on a failure path; the error reports the
  retained locations.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Awaitable, Callable, Dict, List, Optional

from openviking.core.context import ContextLevel
from openviking.server.error_mapping import is_not_found_error
from openviking.server.identity import RequestContext
from openviking.server.skill_package_update import (
    clear_skill_package_contents,
    transfer_skill_package,
)
from openviking.storage.abstract_overview import (
    ABSTRACT_OVERVIEW_FILENAMES,
    rewrite_abstract_overview_for_transfer,
)
from openviking.storage.acl import AclAction
from openviking.storage.errors import LockAcquisitionError, ResourceBusyError
from openviking.storage.internal_names import is_storage_internal_name
from openviking.storage.viking_fs import VikingFS
from openviking_cli.exceptions import (
    InvalidArgumentError,
    OpenVikingError,
    PermissionDeniedError,
    ResourceExhaustedError,
)
from openviking_cli.utils import get_logger

logger = get_logger(__name__)

ABSENT_REVISION = "absent"
# A failed observation must never be used as evidence of absence or safe cleanup.
_UNREADABLE_TREE = object()
REVISION_PREFIX = "tree-sha256:"
# Bound the work done while all three trees are locked.
MAX_TREE_ENTRIES = 20_000
MAX_TREE_BYTES = 512 * 1024 * 1024


def _conflict(message: str, **details: Any) -> OpenVikingError:
    return OpenVikingError(message, code="CONFLICT", details=details)


async def _tree_entries(
    viking_fs: VikingFS, path: str, fs_ctx: Dict[str, Any] | None
) -> List[Dict[str, Any]] | None:
    """Return the visible descendants of ``path`` with their stored bytes' hashes.

    ``None`` means the path does not exist. Storage-internal names such as lock
    tokens are excluded, so a directory created by a Tree lock reads as empty.
    """
    agfs = viking_fs._async_agfs
    try:
        stat = await agfs.stat(path)
    except Exception as exc:
        if is_not_found_error(exc):
            return None
        raise
    if not isinstance(stat, dict) or not stat.get("isDir", False):
        raise InvalidArgumentError("tree revision target must be a directory")
    entries: List[Dict[str, Any]] = []
    total_bytes = 0

    async def visit(directory: str, prefix: str) -> None:
        nonlocal total_bytes
        listing = await agfs.ls(directory, fs_ctx=fs_ctx) if fs_ctx else await agfs.ls(directory)
        for item in sorted(listing, key=lambda value: value.get("name", "")):
            name = item.get("name", "")
            if not name or name in (".", "..") or is_storage_internal_name(name):
                continue
            relative = f"{prefix}{name}"
            child = f"{directory.rstrip('/')}/{name}"
            if len(entries) >= MAX_TREE_ENTRIES:
                raise ResourceExhaustedError(
                    "tree is too large for a guarded replacement",
                    details={"max_entries": MAX_TREE_ENTRIES},
                )
            if item.get("isDir", False):
                entries.append({"path": relative, "type": "dir"})
                await visit(child, f"{relative}/")
                continue
            raw = await agfs.cat(child, fs_ctx=fs_ctx) if fs_ctx else await agfs.cat(child)
            data = raw if isinstance(raw, (bytes, bytearray)) else str(raw).encode("utf-8")
            total_bytes += len(data)
            if total_bytes > MAX_TREE_BYTES:
                raise ResourceExhaustedError(
                    "tree is too large for a guarded replacement",
                    details={"max_bytes": MAX_TREE_BYTES},
                )
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )

    await visit(path, "")
    return entries


async def _expected_transfer_entries(
    viking_fs: VikingFS,
    source_uri: str,
    target_uri: str,
    entries: List[Dict[str, Any]] | None,
    fs_ctx: Dict[str, Any] | None,
    *,
    source_path: str,
) -> List[Dict[str, Any]]:
    """Predict exactly the existing copy helper's destination sidecar bytes."""
    if entries is None:
        raise _conflict("staging tree disappeared before transfer", uri=source_uri)
    expected: List[Dict[str, Any]] = []
    for entry in entries:
        copied = dict(entry)
        relative = entry["path"]
        parent, _, name = relative.rpartition("/")
        if entry["type"] == "file" and name in ABSTRACT_OVERVIEW_FILENAMES:
            raw = await viking_fs._async_agfs.cat(f"{source_path}/{relative}", fs_ctx=fs_ctx)
            data = bytes(raw) if isinstance(raw, (bytes, bytearray)) else str(raw).encode("utf-8")
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise _conflict("staging sidecar changed while predicting transfer", uri=source_uri)
            old_dir = f"{source_uri}/{parent}" if parent else source_uri
            new_dir = f"{target_uri}/{parent}" if parent else target_uri
            rewritten = rewrite_abstract_overview_for_transfer(
                data,
                level=ContextLevel.ABSTRACT if name == ".abstract.md" else ContextLevel.OVERVIEW,
                source_dir_uri=old_dir,
                target_dir_uri=new_dir,
                source_scope_uri=source_uri,
                target_scope_uri=target_uri,
            ).encode("utf-8")
            copied.update(size=len(rewritten), sha256=hashlib.sha256(rewritten).hexdigest())
        expected.append(copied)
    if sum(entry.get("size", 0) for entry in expected) > MAX_TREE_BYTES:
        raise ResourceExhaustedError("rewritten tree exceeds replacement byte limit")
    return expected


def revision_of(entries: List[Dict[str, Any]] | None) -> str:
    """Hash names, types, sizes and stored bytes of every visible descendant.

    Semantic sidecars (``.abstract.md``, ``.overview.md``, ``.relations.json``)
    are files of the tree and so part of the revision. Metadata held outside the
    tree (vector records, tags, ACL rows) is not; it is transferred, not hashed.
    """
    if entries is None:
        return ABSENT_REVISION
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return REVISION_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _summary(entries: List[Dict[str, Any]] | None) -> Dict[str, Any]:
    if entries is None:
        return {"exists": False, "files": 0, "directories": 0, "bytes": 0}
    files = [entry for entry in entries if entry["type"] == "file"]
    return {
        "exists": True,
        "files": len(files),
        "directories": len(entries) - len(files),
        "bytes": sum(entry["size"] for entry in files),
    }


def _validate_revision(value: str, field: str) -> None:
    if value == ABSENT_REVISION:
        return
    digest = value.removeprefix(REVISION_PREFIX)
    if (
        not value.startswith(REVISION_PREFIX)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise InvalidArgumentError(f"{field} must be 'absent' or 'tree-sha256:<hex>'")


async def tree_revision(
    viking_fs: VikingFS, uri: str, *, ctx: RequestContext
) -> Dict[str, Any]:
    """Return a consistent revision of one directory, read under a Tree lease."""
    uri = VikingFS._normalize_transfer_uri(uri).rstrip("/")
    await viking_fs._ensure_access(uri, ctx, action=AclAction.READ)
    path = viking_fs._uri_to_path(uri, ctx=ctx)
    try:
        await viking_fs._async_agfs.stat(path)
    except Exception as exc:
        if is_not_found_error(exc):
            # Locking a missing path would create it; report absence without a lease.
            return {"uri": uri, "revision": ABSENT_REVISION, **_summary(None)}
        raise
    try:
        lease = await viking_fs._async_agfs.pathlock_acquire_tree(path)
    except LockAcquisitionError as exc:
        raise ResourceBusyError(f"tree is busy: {uri}", uri=uri) from exc
    try:
        entries = await _tree_entries(viking_fs, path, viking_fs._pathlock_fs_ctx(ctx, lease))
    finally:
        await viking_fs._async_agfs.pathlock_release(lease)
    return {"uri": uri, "revision": revision_of(entries), **_summary(entries)}


async def replace_tree(
    viking_fs: VikingFS,
    *,
    target_uri: str,
    staging_uri: str,
    backup_uri: str,
    expected_target_revision: str,
    expected_staging_revision: str,
    retain_backup: bool = True,
    ctx: RequestContext,
    fault: Callable[[str], Awaitable[None]] | None = None,
) -> Dict[str, Any]:
    """Replace ``target_uri`` with the contents of ``staging_uri``.

    ``fault`` is a test seam called with each phase name before that phase.
    """
    _validate_revision(expected_target_revision, "expected_target_revision")
    _validate_revision(expected_staging_revision, "expected_staging_revision")
    if expected_staging_revision == ABSENT_REVISION:
        raise InvalidArgumentError("staging must exist")
    uris = [VikingFS._normalize_transfer_uri(uri).rstrip("/") for uri in (target_uri, staging_uri, backup_uri)]
    target_uri, staging_uri, backup_uri = uris
    paths = [viking_fs._uri_to_path(uri, ctx=ctx) for uri in uris]
    target_path, staging_path, backup_path = paths
    for index, first in enumerate(paths):
        for second in paths[index + 1 :]:
            VikingFS._validate_transfer_paths(first, second)
    for uri in uris:
        # Bounded to the shared resources tree, and never its root.
        parts = uri.removeprefix("viking://").split("/")
        if len(parts) < 2 or parts[0] != "resources" or not all(parts):
            raise PermissionDeniedError(
                "tree replacement is limited to directories below viking://resources",
                resource=uri,
            )

    # Destroying the target and consuming staging need MANAGE; the backup needs WRITE.
    await viking_fs._ensure_access(target_uri, ctx, action=AclAction.WRITE)
    await viking_fs._ensure_access(target_uri, ctx, action=AclAction.MANAGE)
    await viking_fs._ensure_access(staging_uri, ctx, action=AclAction.MANAGE)
    await viking_fs._ensure_access(backup_uri, ctx, action=AclAction.WRITE)
    for path, uri in zip(paths, uris):
        await viking_fs._ensure_transfer_parent_directory(path, uri, operation="replace")

    async def exists(path: str) -> bool:
        try:
            await viking_fs._async_agfs.stat(path)
        except Exception as exc:
            if is_not_found_error(exc):
                return False
            raise
        return True

    # Unlocked observations decide which empty directories our own locks create.
    target_existed = await exists(target_path)
    if await exists(backup_path):
        raise _conflict("backup destination already exists", uri=backup_uri)
    if not await exists(staging_path):
        raise _conflict("staging tree no longer exists", uri=staging_uri)

    try:
        lease = await viking_fs._async_agfs.pathlock_acquire_tree_batch(paths)
    except LockAcquisitionError as exc:
        raise ResourceBusyError(f"tree is busy: {target_uri}", uri=target_uri) from exc
    fs_ctx = viking_fs._pathlock_fs_ctx(ctx, lease)

    async def read(path: str) -> List[Dict[str, Any]] | None:
        return await _tree_entries(viking_fs, path, fs_ctx)

    async def remove_placeholder(path: str) -> None:
        entries = await read(path)
        if entries == []:
            await viking_fs._async_agfs.rm(path, recursive=True, fs_ctx=fs_ctx)

    async def step(name: str) -> None:
        if fault is not None:
            await fault(name)

    state: Dict[str, Any] = {
        "target_uri": target_uri,
        "staging_uri": staging_uri,
        "backup_uri": backup_uri,
        "content_committed": False,
    }
    try:
        # -- preconditions under the lease, before any change --------------
        staging_entries = await read(staging_path)
        staging_revision = revision_of(staging_entries)
        target_entries = await read(target_path)
        if target_entries == [] and not target_existed:
            target_entries = None  # our Tree lock created it
        target_revision = revision_of(target_entries)
        backup_entries = await read(backup_path)
        problems = {}
        if staging_revision != expected_staging_revision:
            problems["staging"] = staging_revision
        if target_revision != expected_target_revision:
            problems["target"] = target_revision
        if backup_entries:
            problems["backup"] = "occupied"
        if problems:
            if not target_existed:
                await remove_placeholder(target_path)
            await remove_placeholder(backup_path)
            raise _conflict(
                "tree replacement precondition failed",
                target_uri=target_uri,
                current_revisions=problems,
            )
        state["previous_revision"] = target_revision
        state["revision"] = staging_revision
        expected_installed = revision_of(
            await _expected_transfer_entries(
                viking_fs, staging_uri, target_uri, staging_entries, fs_ctx,
                source_path=staging_path,
            )
        )

        # -- 1. copy the old tree to the backup and verify it --------------
        backup_verified = False
        try:
            await step("backup")
            if target_entries:
                await transfer_skill_package(
                    viking_fs, target_uri, backup_uri, ctx=ctx, lease_ref=lease,
                    preserve_sidecar_bytes=True,
                )
            backup_verified = target_entries is None or (
                revision_of(await read(backup_path)) == target_revision
            )
            if not backup_verified:
                raise RuntimeError("backup copy does not match the target revision")
            # -- 2. clear the target, install staging, verify --------------
            await step("clear_target")
            if target_entries:
                await clear_skill_package_contents(
                    viking_fs, target_uri, ctx=ctx, lease_ref=lease
                )
            elif await read(target_path) is None:
                await viking_fs._async_agfs.mkdir(target_path, fs_ctx=fs_ctx)
            await step("install")
            await transfer_skill_package(
                viking_fs, staging_uri, target_uri, ctx=ctx, lease_ref=lease
            )
            installed = revision_of(await read(target_path))
            if installed != expected_installed:
                raise RuntimeError("installed target does not match the destination-aware manifest")
            state["staging_revision"] = staging_revision
            state["revision"] = installed
        except Exception as failure:
            await _recover(
                viking_fs,
                state=state,
                failure=failure,
                backup_verified=backup_verified,
                target_entries=target_entries,
                read=read,
                paths=(target_path, staging_path, backup_path),
                ctx=ctx,
                lease=lease,
                fs_ctx=fs_ctx,
            )
            raise AssertionError("unreachable")  # _recover always raises
        state["content_committed"] = True

        # -- 3. conditional cleanup ----------------------------------------
        cleanup: Dict[str, Any] = {"status": "completed", "retained": [], "errors": []}
        try:
            await step("cleanup")
            if revision_of(await read(staging_path)) == staging_revision:
                await viking_fs._async_agfs.rm(staging_path, recursive=True, fs_ctx=fs_ctx)
            else:
                cleanup["retained"].append(staging_uri)
            if not target_entries:
                # Nothing was backed up; drop the directory our Tree lock created.
                await remove_placeholder(backup_path)
            elif retain_backup:
                cleanup["retained"].append(backup_uri)
            elif revision_of(await read(backup_path)) == target_revision:
                await viking_fs.rm(backup_uri, recursive=True, ctx=ctx, lease_ref=lease)
            else:
                cleanup["retained"].append(backup_uri)
        except Exception as exc:
            logger.warning("Tree replacement cleanup failed for %s", target_uri, exc_info=True)
            cleanup["status"] = "failed"
            cleanup["errors"].append(type(exc).__name__)
            for uri, path in ((staging_uri, staging_path), (backup_uri, backup_path)):
                if await read(path) and uri not in cleanup["retained"]:
                    cleanup["retained"].append(uri)
        state["cleanup"] = cleanup
        state["backup_retained"] = backup_uri in cleanup["retained"]
        # Vector records moved with the files; nothing was re-embedded. Run the
        # native reindex when staging was written with deferred processing.
        state["index_status"] = "transferred"
        return state
    finally:
        await viking_fs._async_agfs.pathlock_release(lease)


async def _recover(
    viking_fs: VikingFS,
    *,
    state: Dict[str, Any],
    failure: Exception,
    backup_verified: bool,
    target_entries: Optional[List[Dict[str, Any]]],
    read: Callable[[str], Awaitable[Optional[List[Dict[str, Any]]]]],
    paths: tuple[str, str, str],
    ctx: RequestContext,
    lease: Dict[str, Any],
    fs_ctx: Dict[str, Any] | None,
) -> None:
    """Restore the old tree from the verified backup, never deleting a copy blindly."""
    target_path, staging_path, backup_path = paths
    target_uri, staging_uri, backup_uri = (
        state["target_uri"],
        state["staging_uri"],
        state["backup_uri"],
    )
    previous = state["previous_revision"]
    details: Dict[str, Any] = {
        "content_committed": False,
        "target_uri": target_uri,
        "failure": type(failure).__name__,
    }
    current = _reported_revision(await _safe_read(read, target_path))
    if current == "unknown":
        details["state"] = "rollback_unavailable"
        details["observation_failure"] = "target_unreadable"
    elif current == previous or (
        previous == ABSENT_REVISION and current in (ABSENT_REVISION, revision_of([]))
    ):
        # The target files were never changed; any backup is only a copy. The
        # backup transfer may already have moved the target's vector records.
        details["state"] = "target_unchanged"
        backup_entries = await _safe_read(read, backup_path)
        try:
            if backup_entries is _UNREADABLE_TREE:
                details["index_restore_failure"] = "backup_unreadable"
            elif backup_entries:
                await viking_fs._update_vector_store_uris(
                    backup_uri, target_uri, recursive=True, ctx=ctx
                )
            elif backup_entries == []:
                await viking_fs._async_agfs.rm(backup_path, recursive=True, fs_ctx=fs_ctx)
            if previous == ABSENT_REVISION and current == revision_of([]):
                await viking_fs._async_agfs.rm(target_path, recursive=True, fs_ctx=fs_ctx)
        except Exception as exc:
            logger.warning("Tree replacement index restore failed for %s", target_uri, exc_info=True)
            details["index_restore_failure"] = type(exc).__name__
    elif backup_verified or target_entries is None:
        try:
            if previous == ABSENT_REVISION:
                await clear_skill_package_contents(viking_fs, target_uri, ctx=ctx, lease_ref=lease)
            else:
                if revision_of(await read(backup_path)) != previous:
                    raise RuntimeError("backup changed before restoration")
                await clear_skill_package_contents(viking_fs, target_uri, ctx=ctx, lease_ref=lease)
                await transfer_skill_package(
                    viking_fs, backup_uri, target_uri, ctx=ctx, lease_ref=lease,
                    preserve_sidecar_bytes=True,
                )
            restored = revision_of(await read(target_path))
            accepted = (
                (ABSENT_REVISION, revision_of([])) if previous == ABSENT_REVISION else (previous,)
            )
            if restored not in accepted:
                raise RuntimeError("restored target does not match its previous revision")
            details["state"] = "rolled_back"
        except Exception as exc:
            logger.error("Tree replacement rollback failed for %s", target_uri, exc_info=True)
            details["state"] = "rollback_failed"
            details["rollback_failure"] = type(exc).__name__
    else:
        details["state"] = "rollback_unavailable"
    retained = []
    for uri, path in ((backup_uri, backup_path), (staging_uri, staging_path)):
        entries = await _safe_read(read, path)
        if entries is not None:
            retained.append({"uri": uri, "revision": _reported_revision(entries)})
    details["retained"] = retained
    details["target_revision"] = _reported_revision(await _safe_read(read, target_path))
    # File content is restored from copies; index records may need the native reindex.
    details["index_status"] = "reindex_recommended"
    code = "INTERNAL" if details["state"] in ("rolled_back", "target_unchanged") else "DATA_LOSS"
    raise OpenVikingError(
        f"tree replacement failed; {details['state']}",
        code=code,
        details=details,
    ) from failure


async def _safe_read(read, path):
    try:
        return await read(path)
    except Exception:
        return _UNREADABLE_TREE


def _reported_revision(entries):
    return "unknown" if entries is _UNREADABLE_TREE else revision_of(entries)
