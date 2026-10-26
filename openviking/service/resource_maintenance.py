# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Resource maintenance without extraction, refresh, watch, version or task effects."""

from openviking.server.error_mapping import is_not_found_error, map_exception
from openviking.storage.acl import AclAction
from openviking.storage.internal_names import is_storage_internal_name
from openviking.utils.path_safety import validate_safe_viking_uri_path
from openviking_cli.exceptions import (
    ConflictError, FailedPreconditionError, InvalidArgumentError, NotFoundError,
    OpenVikingError,
)


def validate_resource_target(uri: str) -> str:
    """Accept only exact, concrete resources paths; never resolve aliases or IDs."""
    if not isinstance(uri, str) or not 1 <= len(uri) <= 2048:
        raise InvalidArgumentError("A concrete resources URI of at most 2048 characters is required")
    if not uri.startswith("viking://resources/"):
        raise InvalidArgumentError("A concrete path below viking://resources is required")
    parts = uri[len("viking://resources/"):].split("/")
    if any(not part or part in {".", ".."} or is_storage_internal_name(part) for part in parts):
        raise InvalidArgumentError("Empty, relative and storage-internal path segments are not allowed")
    # Reject alternate spellings rather than silently broadening the target.
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in uri) or any(
        c in uri for c in ("\\", "%", "?", "#", "{", "}")
    ):
        raise InvalidArgumentError("An exact literal resources URI is required")
    try:
        if validate_safe_viking_uri_path(uri) != uri:
            raise ValueError("noncanonical URI")
    except ValueError as exc:
        raise InvalidArgumentError("An exact literal resources URI is required") from exc
    return uri


async def maintain_resource(viking_fs, uri, ctx, *, recursive=False, prune=False):
    """Remove a present node, or prune exact-URI records for an absent node.

    Absence and native ACL checks run under account-checked resources-root Tree
    coverage. rm extends that same owner lease and performs its own fresh checks.
    Prune deliberately has no recursive mode: absent descendants cannot be
    discovered or ACL-qualified by the filesystem-only rm collector.
    """
    uri = validate_resource_target(uri)
    effects_started = False
    try:
        await viking_fs._ensure_access(uri, ctx, action=AclAction.MANAGE)
        await viking_fs._ensure_access(uri, ctx, action=AclAction.WRITE)
        admitted = await viking_fs._capture_resource_identity(ctx)
        lease = await viking_fs._acquire_resource_coverage(ctx, admitted)
        try:
            await viking_fs._ensure_access(uri, ctx, action=AclAction.MANAGE)
            await viking_fs._ensure_access(uri, ctx, action=AclAction.WRITE)
            path = viking_fs._uri_to_path(uri, ctx=ctx)
            try:
                fresh = await viking_fs._async_agfs.stat(path)
            except Exception as exc:
                if not is_not_found_error(exc):
                    raise
                fresh = None
            if prune:
                if fresh is not None:
                    raise ConflictError("Orphan pruning requires an absent resource")
                if viking_fs._get_vector_store() is None:
                    raise FailedPreconditionError("Resource index is unavailable")
            else:
                if fresh is None:
                    raise NotFoundError(uri)
                if fresh.get("isDir", False) and not recursive:
                    raise InvalidArgumentError("Directory removal requires recursive=true")
            effects_started = True
            await viking_fs.rm(
                uri, ctx=ctx, recursive=False if prune else recursive, lease_ref=lease,
            )
        finally:
            await viking_fs._async_agfs.pathlock_release(lease)
    except Exception as exc:
        # Never pass backend counts, index contents, foreign lease owners or raw
        # storage exceptions to the caller. Partial deletion is not rolled back.
        mapped = exc if isinstance(exc, OpenVikingError) else map_exception(exc, resource=uri)
        code = getattr(mapped, "code", "INTERNAL")
        allowed = {
            "INVALID_ARGUMENT", "INVALID_URI", "PERMISSION_DENIED", "NOT_FOUND",
            "CONFLICT", "FAILED_PRECONDITION", "RESOURCE_BUSY", "RESOURCE_EXHAUSTED",
        }
        if code not in allowed:
            code = "INTERNAL"
        raise OpenVikingError(
            "Resource maintenance did not complete",
            code=code,
            details={"uri": uri, "outcome": "unknown" if effects_started else "not_started"},
        ) from exc
    return {"uri": uri, "outcome": "pruned" if prune else "removed"}
