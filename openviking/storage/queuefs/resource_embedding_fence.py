# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Resource-only preconditions for the persisted embedding commit boundary."""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime

from openviking.server.account_stores.models import AccountIdentity
from openviking.server.identity import RequestContext
from openviking.storage.abstract_overview import body_for_preview
from openviking.storage.errors import LockAcquisitionError
from openviking.storage.index_action import IndexAction
from openviking.storage.pathlock_transfer import PathLockGuard
from openviking.storage.vector_ids import vector_record_id
from openviking.utils.content_hash import content_md5


class ResourceEmbeddingConflict(ValueError):
    """A resource embedding no longer has qualified original inputs."""


@dataclass(frozen=True)
class ResourceEmbeddingProducer:
    ctx: RequestContext
    identity: AccountIdentity
    owner: dict


_producer: ContextVar[ResourceEmbeddingProducer | None] = ContextVar(
    "resource_embedding_producer", default=None
)


def get_resource_embedding_producer():
    return _producer.get()


@contextmanager
def bind_resource_embedding_producer(producer):
    token = _producer.set(producer)
    try:
        yield
    finally:
        _producer.reset(token)


def _resource_uri(uri):
    return isinstance(uri, str) and (
        uri.rstrip("/") == "viking://resources" or uri.startswith("viking://resources/")
    )


def is_resource_embedding(msg):
    uri = msg.context_data.get("uri")
    if uri:
        return _resource_uri(uri)
    seed = msg.field_patch.seed_fields if msg.field_patch is not None else {}
    return _resource_uri(seed.get("uri")) or msg.context_data.get("context_type") == "resource"


def _json_value(value):
    if isinstance(value, (date, datetime)):
        return {"datetime": value.isoformat(), "type": type(value).__name__}
    raise TypeError("Unsupported resource context revision value")


def _record_hash(records):
    return hashlib.sha256(
        json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_value,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _binding(msg):
    data = msg.context_data
    if msg.action not in {IndexAction.UPSERT, IndexAction.MERGE}:
        raise ResourceEmbeddingConflict("Unsupported cached resource index action")
    uri, account, level = data.get("uri"), data.get("account_id"), data.get("level")
    if not is_resource_embedding(msg) or type(level) is not int or level not in (0, 1, 2):
        raise ResourceEmbeddingConflict("Unsupported resource embedding input")
    record_id = data.get("_upsert_record_id") or vector_record_id(account, uri, level)
    source_uri = (
        uri
        if level == 2
        else uri.rstrip("/") + ("/.abstract.md" if level == 0 else "/.overview.md")
    )
    return account, uri, level, str(record_id), source_uri


async def _records(db, record_id, uri, account, ctx):
    records = await db.get_strict([record_id], ctx=ctx)
    if any(record.get("uri") != uri or record.get("account_id") != account for record in records):
        raise ResourceEmbeddingConflict("Resource index record binding changed")
    return records


async def stamp_resource_embedding(msg):
    """Stamp only verified semantic producers, never arbitrary cached messages."""
    if not is_resource_embedding(msg) or msg.action is IndexAction.NONE:
        return
    if "_resource_fence" in msg.context_data:
        # Retries must retain the original condition rather than bless new input.
        return
    producer = _producer.get()
    if producer is None:
        raise ResourceEmbeddingConflict("Resource embedding producer has no owned input scope")
    from openviking.service.account_identity import recheck_account_identity
    from openviking.storage.viking_fs import get_viking_fs

    fs = get_viking_fs()
    account, uri, level, record_id, source_uri = _binding(msg)
    if account != producer.ctx.account_id:
        raise ResourceEmbeddingConflict("Resource embedding account binding changed")
    if await fs._async_agfs.pathlock_refresh(producer.owner) != "refreshed":
        raise ResourceEmbeddingConflict("Resource embedding producer lost ownership")
    coverage = await fs._async_agfs.pathlock_to_handoff(producer.owner)
    path = fs._uri_to_path(source_uri, ctx=producer.ctx)
    if not any(
        item.get("kind") == "tree"
        and (path == item["path"].rstrip("/") or path.startswith(item["path"].rstrip("/") + "/"))
        for item in coverage.get("covered_paths", [])
    ):
        raise ResourceEmbeddingConflict("Resource embedding producer lacks input tree coverage")
    await recheck_account_identity(fs._account_identity_manager, account, producer.identity)
    raw = await fs.read_file_bytes(source_uri, ctx=producer.ctx)
    expected = msg.context_data.get("md5")
    comparable = raw if level == 2 else body_for_preview(raw).encode()
    if not isinstance(expected, str) or not expected or content_md5(comparable) != expected:
        raise ResourceEmbeddingConflict("Resource embedding input lacks a matching byte condition")
    records = await _records(fs._get_vector_store(), record_id, uri, account, producer.ctx)
    msg.context_data["_resource_fence"] = {
        "account_id": account,
        "creation_identity": producer.identity.creation_identity,
        "uri": uri,
        "level": level,
        "record_id": record_id,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "context_sha256": _record_hash(records),
        "owner": dict(producer.owner),
    }


@asynccontextmanager
async def resource_embedding_commit(msg, ctx, db):
    """Recheck original account, bytes and context under final covering ownership."""
    if not is_resource_embedding(msg) or msg.action is IndexAction.NONE:
        if msg.action is not IndexAction.NONE and not msg.context_data.get("uri"):
            ids = msg.record_ids or [
                msg.context_data.get("_upsert_record_id") or msg.context_data.get("id")
            ]
            records = await db.get_strict([value for value in ids if value], ctx=ctx)
            if any(_resource_uri(record.get("uri")) for record in records):
                raise ResourceEmbeddingConflict("Unbound legacy resource index action")
        yield
        return
    from openviking.storage.viking_fs import get_viking_fs

    account, uri, level, record_id, source_uri = _binding(msg)
    proof = msg.context_data.pop("_resource_fence", None)
    if not isinstance(proof, dict) or (
        proof.get("account_id"),
        proof.get("uri"),
        proof.get("level"),
        proof.get("record_id"),
    ) != (account, uri, level, record_id):
        raise ResourceEmbeddingConflict("Legacy resource embedding lacks original input conditions")
    identity = proof.get("creation_identity")
    if not isinstance(identity, str) or not identity:
        raise ResourceEmbeddingConflict("Resource embedding lacks original account identity")
    fs = get_viking_fs()
    owner = proof.get("owner")
    if not isinstance(owner, dict):
        raise ResourceEmbeddingConflict("Resource embedding lacks original ownership")
    try:
        status = await fs._async_agfs.pathlock_refresh(owner)
    except (LockAcquisitionError, RuntimeError, ValueError) as error:
        if "owned lease capability does not match ref" not in str(error):
            raise
        owner = None
    else:
        if status == "lost":
            owner = None
        elif status != "refreshed":
            raise ResourceEmbeddingConflict("Resource embedding ownership cannot be validated")
    # D3 rechecks the authoritative creation identity/deleting flag AFTER this
    # final acquisition. Reentrant extension keeps a live semantic owner intact.
    lease = await fs._acquire_resource_coverage(
        ctx, AccountIdentity(identity, False), owner_lease_ref=owner
    )
    guard = PathLockGuard(fs._async_agfs, lease)
    try:
        raw = await fs.read_file_bytes(source_uri, ctx=ctx)
        if hashlib.sha256(raw).hexdigest() != proof.get("input_sha256"):
            raise ResourceEmbeddingConflict("Resource embedding source bytes changed")
        records = await _records(db, record_id, uri, account, ctx)
        if _record_hash(records) != proof.get("context_sha256"):
            raise ResourceEmbeddingConflict("Resource embedding context revision changed")
        if await fs._async_agfs.pathlock_refresh(lease) != "refreshed":
            raise ResourceEmbeddingConflict("Resource embedding final ownership was lost")
        yield
    finally:
        await guard.close()
