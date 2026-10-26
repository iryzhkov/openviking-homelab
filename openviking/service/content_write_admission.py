"""Original request admission for resource content-write jobs."""

from contextlib import asynccontextmanager

from openviking.service.account_identity import AccountIdentity, recheck_account_identity
from openviking.storage.pathlock_transfer import PathLockGuard
from openviking.storage.queuefs.resource_embedding_fence import (
    ResourceEmbeddingConflict,
    _resource_uri,
    bind_resource_index_conditions,
    bind_resource_semantic_admission,
)


def content_write_admission(msg):
    """Read carried admission, never capture identity from a queued request."""
    if not _resource_uri(msg.resource_id):
        return None
    kind = str(msg.refresh.get("kind") or "none")
    if kind == "none":
        return None
    if kind != "plan":
        # Batch and other unbound refresh shapes would write freshness or
        # sidecars before any original-identity check. Refuse at admission.
        raise ResourceEmbeddingConflict("Unbound resource content refresh is unsupported")
    proof = msg.refresh.get("resource_admission")
    if not isinstance(proof, dict):
        raise ResourceEmbeddingConflict("Resource content write lacks original admission")
    identity = proof.get("creation_identity")
    if not isinstance(identity, str) or not identity:
        raise ResourceEmbeddingConflict("Resource content write lacks original identity")
    if not isinstance(proof.get("owner"), dict) or msg.lock_handoff is None:
        raise ResourceEmbeddingConflict("Resource content write lacks original owner")
    return proof


async def recheck_content_write_admission(fs, msg, ctx):
    proof = content_write_admission(msg)
    if proof is not None:
        await recheck_account_identity(
            fs._account_identity_manager,
            ctx.account_id,
            AccountIdentity(proof["creation_identity"], False),
        )
    return proof


@asynccontextmanager
async def bind_content_write_admission(fs, msg, ctx, owner):
    """Recheck the adopted original owner before dispatching carried work."""
    proof = await recheck_content_write_admission(fs, msg, ctx)
    if proof is None:
        yield
        return
    # Native adoption rotates the owned capability. The consumer reaches
    # this point only by adopting this message's original handoff, with no
    # reacquisition fallback. Index conditions retain their capture-time owner.
    if owner is None:
        raise ResourceEmbeddingConflict("Resource content write has no adopted owner")
    if await fs._async_agfs.pathlock_refresh(owner) != "refreshed":
        raise ResourceEmbeddingConflict("Resource content write lost original owner")
    coverage = await fs._acquire_resource_coverage(
        ctx, AccountIdentity(proof["creation_identity"], False), owner_lease_ref=owner
    )
    guard = PathLockGuard(fs._async_agfs, coverage)
    try:
        # Release broad coverage before semantic/embedding work starts.
        # The original input owner remains live through dispatch.
        await guard.close()
        with bind_resource_index_conditions(proof.get("index_conditions"), proof["owner"]):
            with bind_resource_semantic_admission(ctx.account_id, proof["creation_identity"]):
                yield
    finally:
        await guard.close()
