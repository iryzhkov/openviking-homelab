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
    index_actions: bool = False


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
    if msg.action not in {
        IndexAction.UPSERT,
        IndexAction.MERGE,
        IndexAction.UPDATE_FIELDS,
        IndexAction.DELETE,
    }:
        raise ResourceEmbeddingConflict("Unsupported cached resource index action")
    uri, account, level = data.get("uri"), data.get("account_id"), data.get("level")
    if not is_resource_embedding(msg) or type(level) is not int or level not in (0, 1, 2):
        raise ResourceEmbeddingConflict("Unsupported resource embedding input")
    if msg.action in {IndexAction.UPDATE_FIELDS, IndexAction.DELETE}:
        if len(msg.record_ids) != 1:
            raise ResourceEmbeddingConflict(
                "Resource index actions require one exact record binding"
            )
        record_id = msg.record_ids[0]
    else:
        record_id = data.get("_upsert_record_id") or vector_record_id(account, uri, level)
    source_uri = (
        uri
        if level == 2
        else uri.rstrip("/") + ("/.abstract.md" if level == 0 else "/.overview.md")
    )
    return account, uri, level, str(record_id), source_uri


async def _records(db, record_id, uri, account, ctx, level=None):
    records = await db.get_strict([record_id], ctx=ctx)
    if any(
        record.get("uri") != uri
        or record.get("account_id") != account
        or (level is not None and record.get("level") != level)
        for record in records
    ):
        raise ResourceEmbeddingConflict("Resource index record binding changed")
    return records


_index_conditions: ContextVar[tuple | None] = ContextVar("resource_index_conditions", default=None)


@contextmanager
def bind_resource_index_conditions(conditions, owner=None):
    """Carry already captured conditions; never mint a new revision for cached work."""
    token = _index_conditions.set((conditions, owner))
    try:
        yield
    finally:
        _index_conditions.reset(token)


async def _input_state(fs, uri, source_uri, ctx):
    from openviking.server.error_mapping import is_not_found_error

    try:
        info = await fs.stat(uri, ctx=ctx, skip_count=True)
    except Exception as error:
        if not is_not_found_error(error):
            raise
        return {"kind": "absent", "sha256": None}
    kind = "directory" if info.get("isDir") else "file"
    try:
        raw = await fs.read_file_bytes(source_uri, ctx=ctx)
    except Exception as error:
        if not is_not_found_error(error) and not isinstance(error, IsADirectoryError):
            raise
        digest = None
    else:
        digest = hashlib.sha256(raw).hexdigest()
    return {"kind": kind, "sha256": digest}


async def stamp_resource_embedding(msg):
    """Only current owned producers or immutable original conditions may issue work."""
    if not is_resource_embedding(msg) or msg.action is IndexAction.NONE:
        return
    if "_resource_fence" in msg.context_data:
        return
    account, uri, level, record_id, source_uri = _binding(msg)
    carried = _index_conditions.get()
    if carried is not None:
        conditions, owner = carried
        proof = (conditions or {}).get(f"{msg.action.value}:{record_id}")
        if not isinstance(proof, dict) or (
            proof.get("account_id"),
            proof.get("uri"),
            proof.get("level"),
            proof.get("record_id"),
            proof.get("action"),
        ) != (account, uri, level, record_id, msg.action.value):
            raise ResourceEmbeddingConflict("Index action lacks its original captured conditions")
        if msg.action is IndexAction.UPDATE_FIELDS and proof.get("intent_sha256") != _record_hash(
            msg.field_patch.to_dict()
        ):
            raise ResourceEmbeddingConflict("Resource patch intent changed after admission")
        # The captured owner is part of the condition; a later or different
        # owner must never inherit conditions captured under another lease.
        if owner is None or proof.get("owner") != dict(owner):
            raise ResourceEmbeddingConflict("Index action conditions belong to another owner")
        if not isinstance(proof.get("creation_identity"), str) or not proof["creation_identity"]:
            raise ResourceEmbeddingConflict("Index action lacks original account identity")
        msg.context_data["_resource_fence"] = dict(proof)
        return
    producer = _producer.get()
    if producer is None:
        raise ResourceEmbeddingConflict("Resource embedding producer has no owned input scope")
    from openviking.storage.viking_fs import get_viking_fs

    fs = get_viking_fs()
    if account != producer.ctx.account_id:
        raise ResourceEmbeddingConflict("Resource embedding account binding changed")
    if await fs._async_agfs.pathlock_refresh(producer.owner) != "refreshed":
        raise ResourceEmbeddingConflict("Resource embedding producer lost ownership")
    coverage = await fs._async_agfs.pathlock_to_handoff(producer.owner)
    path = fs._uri_to_path(uri, ctx=producer.ctx)
    if not any(
        (
            item.get("kind") == "tree"
            and (
                path == item["path"].rstrip("/") or path.startswith(item["path"].rstrip("/") + "/")
            )
        )
        or (level == 2 and item.get("kind") == "exact" and path == item["path"])
        for item in coverage.get("covered_paths", [])
    ):
        raise ResourceEmbeddingConflict("Resource embedding producer lacks input coverage")
    lease = await fs._acquire_resource_coverage(
        producer.ctx, producer.identity, owner_lease_ref=producer.owner
    )
    guard = PathLockGuard(fs._async_agfs, lease)
    try:
        records = await _records(
            fs._get_vector_store(), record_id, uri, account, producer.ctx, level
        )
        proof = {
            "account_id": account,
            "creation_identity": producer.identity.creation_identity,
            "uri": uri,
            "level": level,
            "record_id": record_id,
            "action": msg.action.value,
            "context_sha256": _record_hash(records),
            "owner": dict(producer.owner),
        }
        if msg.action in {IndexAction.UPDATE_FIELDS, IndexAction.DELETE}:
            if not producer.index_actions:
                raise ResourceEmbeddingConflict("Resource index intent lacks original admission")
            state = await _input_state(fs, uri, source_uri, producer.ctx)
            if msg.action is IndexAction.UPDATE_FIELDS and (
                not records or state["kind"] == "absent"
            ):
                raise ResourceEmbeddingConflict(
                    "Resource patch cannot create a missing context or source"
                )
            if msg.action is IndexAction.UPDATE_FIELDS:
                if set(msg.field_patch.values) - {"md5", "search_tags"}:
                    raise ResourceEmbeddingConflict("Cached model-output patches are unsupported")
                if "md5" in msg.field_patch.values:
                    raw = await fs.read_file_bytes(source_uri, ctx=producer.ctx)
                    comparable = raw if level == 2 else body_for_preview(raw).encode()
                    if content_md5(comparable) != msg.field_patch.values["md5"]:
                        raise ResourceEmbeddingConflict("Resource patch fingerprint is stale")
                proof["intent_sha256"] = _record_hash(msg.field_patch.to_dict())
            proof["input_state"] = state
        else:
            raw = await fs.read_file_bytes(source_uri, ctx=producer.ctx)
            expected = msg.context_data.get("md5")
            comparable = raw if level == 2 else body_for_preview(raw).encode()
            if not isinstance(expected, str) or not expected or content_md5(comparable) != expected:
                raise ResourceEmbeddingConflict(
                    "Resource embedding input lacks a matching byte condition"
                )
            proof["input_sha256"] = hashlib.sha256(raw).hexdigest()
        msg.context_data["_resource_fence"] = proof
    finally:
        await guard.close()


async def capture_resource_index_conditions(actions, ctx, admitted, owner):
    """Call while the original planning/content owner is continuously live."""
    from openviking.storage.queuefs.embedding_msg import EmbeddingMsg

    result = {}
    with bind_resource_embedding_producer(ResourceEmbeddingProducer(ctx, admitted, owner, True)):
        for action in actions:
            if not _resource_uri(action.uri):
                continue
            data = {"uri": action.uri, "level": action.level, "account_id": ctx.account_id}
            if action.action is IndexAction.DELETE:
                msg = EmbeddingMsg.for_delete(record_ids=[action.record_id], context_data=data)
            elif action.action is IndexAction.UPDATE_FIELDS:
                msg = EmbeddingMsg.for_update_fields(
                    record_id=action.record_id, context_data=data, field_patch=action.field_patch
                )
            else:
                data.update(md5=action.md5, _upsert_record_id=action.record_id)
                msg = EmbeddingMsg.for_embed(
                    message="capture", context_data=data, action=action.action
                )
            await stamp_resource_embedding(msg)
            result[f"{action.action.value}:{action.record_id}"] = msg.context_data[
                "_resource_fence"
            ]
    return result


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
    if (
        msg.action in {IndexAction.UPDATE_FIELDS, IndexAction.DELETE}
        and proof.get("action") != msg.action.value
    ):
        raise ResourceEmbeddingConflict("Resource index action binding changed")
    if msg.action is IndexAction.UPDATE_FIELDS and proof.get("intent_sha256") != _record_hash(
        msg.field_patch.to_dict()
    ):
        raise ResourceEmbeddingConflict("Resource patch intent changed after admission")
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
        if msg.action in {IndexAction.UPDATE_FIELDS, IndexAction.DELETE}:
            state = await _input_state(fs, uri, source_uri, ctx)
            if state != proof.get("input_state"):
                raise ResourceEmbeddingConflict("Resource index input state changed")
        else:
            raw = await fs.read_file_bytes(source_uri, ctx=ctx)
            if hashlib.sha256(raw).hexdigest() != proof.get("input_sha256"):
                raise ResourceEmbeddingConflict("Resource embedding source bytes changed")
        records = await _records(db, record_id, uri, account, ctx, level)
        if _record_hash(records) != proof.get("context_sha256"):
            raise ResourceEmbeddingConflict("Resource embedding context revision changed")
        if await fs._async_agfs.pathlock_refresh(lease) != "refreshed":
            raise ResourceEmbeddingConflict("Resource embedding final ownership was lost")
        yield
    finally:
        await guard.close()
