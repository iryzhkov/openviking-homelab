"""Closed resource document publication over private parser artifacts.

No import/watch controls and no persisted journal. Rollback is same-process only.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openviking.core.context import Context, ContextLevel
from openviking.parse.accessors.base import LocalResource, SourceType
from openviking.parse.output import (
    LocalParseOutputStore,
    ParseArtifactWriter,
    resolve_artifact_doc_root,
)
from openviking.server.error_mapping import is_not_found_error
from openviking.service import tree_replace
from openviking.service.account_identity import recheck_account_identity
from openviking.storage.abstract_overview import (
    ABSTRACT_OVERVIEW_FILENAMES,
    rewrite_abstract_overview_for_transfer,
)
from openviking.storage.acl import AclAction, AclMode
from openviking.storage.context_update_plan import (
    build_context_update_plan_from_snapshot,
    execute_content_tree_actions,
)
from openviking.storage.expr import And, Contains, Eq, Or, PathScope
from openviking.storage.internal_names import is_storage_internal_name
from openviking.storage.queuefs import QueueManager
from openviking.storage.queuefs.add_resource_msg import AddResourceMsg, AddResourcePhase
from openviking.storage.queuefs.resource_embedding_fence import capture_resource_index_conditions
from openviking.storage.resource_diff import build_rnfv_snapshot, prepare_artifact_inventory
from openviking.storage.resource_rfv import (
    validate_resource_document_snapshot_uri,
    validate_resource_reindex_uri,
)
from openviking.storage.resource_rnfv import CONTROL_BASENAMES, RequestIntent
from openviking.storage.resource_target import AgfsResourceTarget
from openviking.storage.vector_ids import vector_record_id
from openviking.utils.ingest_options import IngestOptions
from openviking.utils.resource_processor import _DocRelStore
from openviking_cli.exceptions import (
    ConflictError,
    FailedPreconditionError,
    InvalidArgumentError,
    OpenVikingError,
    PermissionDeniedError,
    ResourceExhaustedError,
)
from openviking_cli.utils.config import get_openviking_config

MAX_TEXT_BYTES = 1024 * 1024
MAX_REASON_BYTES = 4096
FORMAL_LIMIT = 128
RECORD_LIMIT = 384
ACTION_LIMIT = 384
CONTROL_PREFIX = "file-controls-v1-sha256:"


class _DocumentIntent(RequestIntent):
    def required_vector_fields(self):
        return super().required_vector_fields() | {"search_tags", "tags"}


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AbsentNode(_Closed):
    kind: Literal["absent"]


class FileNode(_Closed):
    kind: Literal["file"]
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    control_revision: str = Field(pattern=r"^file-controls-v1-sha256:[a-f0-9]{64}$")


class DirectoryNode(_Closed):
    kind: Literal["directory"]
    tree_revision: str = Field(pattern=r"^tree-sha256:[a-f0-9]{64}$")


Node = Annotated[AbsentNode | FileNode | DirectoryNode, Field(discriminator="kind")]


class DocumentWriteRequest(_Closed):
    target_uri: str
    source_name: str
    text: str
    reason: str = ""
    representation: Literal["ingest", "raw_file", "verbatim", "exact_directory"] = "ingest"
    contract_version: Literal[1, 2] = 1
    expected: Node
    backup_uri: str | None = None

    @model_validator(mode="after")
    def closed_shape(self):
        try:
            size, reason_size = len(self.text.encode("utf-8")), len(self.reason.encode("utf-8"))
        except UnicodeEncodeError:
            raise ValueError("Document input must be valid UTF-8") from None
        if size > MAX_TEXT_BYTES or reason_size > MAX_REASON_BYTES:
            raise ValueError("Document input limit exceeded")
        if not re.fullmatch(r"[^/\\\x00-\x1f\x7f]+", self.source_name) or self.source_name in {
            ".",
            "..",
        }:
            raise ValueError("source_name must be a safe input leaf")
        if len(self.source_name.encode("utf-8")) > 255:
            raise ValueError("source_name limit exceeded")
        if (self.expected.kind == "absent") != (self.backup_uri is None):
            raise ValueError("backup_uri is required exactly for replacement")
        if self.representation == "raw_file" and self.expected.kind != "file":
            raise ValueError("raw_file requires an existing file expectation")
        if self.representation == "verbatim" and not self.target_uri.endswith(".verbatim"):
            raise ValueError("verbatim requires a .verbatim target")
        if self.representation == "exact_directory" and (
            self.contract_version != 2 or self.expected.kind != "directory"
        ):
            raise ValueError("exact_directory requires contract v2 and an existing directory")
        return self


@dataclass
class _Snapshot:
    node: dict
    entries: list
    records: list
    bodies: dict


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _uri(fs, uri, ctx, *, generated=False):
    try:
        if generated:
            validate_resource_document_snapshot_uri(fs, uri, ctx)
        else:
            validate_resource_reindex_uri(fs, uri, ctx)
        if not generated and any(part in CONTROL_BASENAMES for part in uri.split("/")[3:]):
            raise ValueError()
        if any(is_storage_internal_name(part) for part in uri.split("/")[3:]):
            raise ValueError()
    except ValueError:
        raise InvalidArgumentError("Document URI must be canonical resources content") from None
    return uri


async def _records(fs, uri, ctx):
    """Bounded strict account-scoped inventory, including fragment controls."""
    store = fs._get_vector_store()
    if store is None:
        raise FailedPreconditionError("Document context store unavailable")
    guard = replace(ctx, bypass_acl=True)
    condition = And(
        [
            Eq("account_id", ctx.account_id),
            Or([Eq("uri", uri), PathScope("uri", uri, depth=-1), Contains("uri", uri + "#")]),
        ]
    )
    count = await store._strict_transfer_count(guard, condition)
    if count > RECORD_LIMIT:
        raise ResourceExhaustedError("Document context limit exceeded")
    page, _ = await store._strict_transfer_page(
        guard, condition, limit=RECORD_LIMIT + 1, cursor=None, output_fields=["id", "uri"]
    )
    if len(page) != count:
        raise FailedPreconditionError("Document context inventory incomplete")
    ids = [str(row["id"]) for row in page]
    rows = await store._strict_transfer_get(guard, ids) if ids else []
    if len(rows) != count or any(
        row.get("account_id") != ctx.account_id
        or not (
            row.get("uri") == uri
            or str(row.get("uri", "")).startswith(uri + "/")
            or str(row.get("uri", "")).startswith(uri + "#")
        )
        for row in rows
    ):
        raise FailedPreconditionError("Document context inventory incomplete")
    await tree_replace._require_tree_access(
        fs, uri, [row["uri"] for row in rows], ctx, AclAction.READ
    )
    return deepcopy(rows)


def _descriptor(uri, entries, records, bodies, kind):
    if kind == "absent":
        return {"kind": "absent"}
    if kind == "directory":
        return {
            "kind": kind,
            "tree_revision": tree_replace.revision_of(
                tree_replace._with_metadata(entries, records, uri)
            ),
        }
    controls = {"version": 1, "records": tree_replace._metadata_entry(records, uri)}
    return {
        "kind": "file",
        "content_sha256": hashlib.sha256(bodies[""]).hexdigest(),
        "control_revision": CONTROL_PREFIX + _digest(controls),
    }


async def _snapshot(fs, uri, ctx, lease, *, owned_absence=False):
    try:
        return await _snapshot_inputs(fs, uri, ctx, lease, owned_absence=owned_absence)
    except PermissionDeniedError:
        pass
    # Raise outside the handler so even the exception context cannot name a child.
    raise PermissionDeniedError("Tree contains inaccessible entries", resource=uri)


async def _snapshot_inputs(fs, uri, ctx, lease, *, owned_absence=False):
    await fs._ensure_access(uri, ctx, action=AclAction.READ)
    records = await _records(fs, uri, ctx)
    path = fs._uri_to_path(uri, ctx=ctx)
    try:
        st = await fs._async_agfs.stat(path)
    except Exception as exc:
        if not is_not_found_error(exc):
            raise
        if records and not owned_absence:
            raise FailedPreconditionError("Resource document has residual index state")
        return _Snapshot({"kind": "absent"}, [], records, {})
    if st.get("isSymlink") or st.get("is_symlink"):
        raise FailedPreconditionError("Document aliases are unsupported")
    kind = "directory" if st.get("isDir") else "file"
    if kind == "directory":
        if any("#" in row["uri"][len(uri) :] for row in records):
            raise FailedPreconditionError("Directory fragment controls are unsupported")
        entries = await tree_replace._tree_entries(
            fs, path, fs._pathlock_fs_ctx(ctx, lease), ctx=ctx, max_entries=FORMAL_LIMIT - 1
        )
        entries = [row for row in entries if row["type"] != "metadata"]
        bodies = {}
        for row in entries:
            child = uri + "/" + row["path"]
            _uri(fs, child, ctx, generated=True)
            if row["type"] == "file":
                bodies[row["path"]] = await fs.read_file_bytes(child, ctx=ctx)
    else:
        raw = await fs.read_file_bytes(uri, ctx=ctx)
        if len(raw) > tree_replace.MAX_TREE_BYTES:
            raise ResourceExhaustedError("Document byte limit exceeded")
        bodies = {"": raw}
        entries = [
            {
                "path": "",
                "type": "file",
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ]
    return _Snapshot(_descriptor(uri, entries, records, bodies, kind), entries, records, bodies)


async def document_node(service, uri, ctx):
    fs = service.viking_fs
    _uri(fs, uri, ctx)
    admitted = await fs._capture_resource_identity(ctx)
    lease = await fs._acquire_resource_coverage(ctx, admitted)
    try:
        await _live(fs, ctx, admitted, lease)
        return {"uri": uri, "node": (await _snapshot(fs, uri, ctx, lease)).node}
    finally:
        await fs._async_agfs.pathlock_release(lease)


async def _live(fs, ctx, admitted, lease):
    if await fs._async_agfs.pathlock_refresh(lease) != "refreshed":
        raise ConflictError("Document ownership was lost")
    await recheck_account_identity(fs._account_identity_manager, ctx.account_id, admitted)


def _root_record(uri, level, ctx, is_dir):
    record = Context(
        uri=uri,
        level=ContextLevel(level),
        is_leaf=not is_dir,
        context_type="resource",
        account_id=ctx.account_id,
        user=ctx.user,
    ).to_dict()
    record["id"] = vector_record_id(ctx.account_id, uri, level)
    return record


def _controls_for_level(old):
    controls = tree_replace._controls(old)
    permitted = {"acl_mode", "acl_direct_grants", "search_tags", "tags"}
    ignored = {"parent_uri", "temp_uri", "is_leaf", "user"}
    if any(
        key not in permitted | ignored and value not in (None, "", [], {}, False, 0)
        for key, value in controls.items()
    ):
        raise FailedPreconditionError("Document root controls cannot change level")
    return {key: value for key, value in controls.items() if key in permitted}


def _validate_acl_snapshots(records):
    policies = {}
    slots = {}
    for row in records:
        fields = tree_replace._controls(row)
        policy = (fields["acl_mode"], tuple(fields["acl_direct_grants"]))
        if row["uri"] in policies and policies[row["uri"]] != policy:
            raise FailedPreconditionError("Document controls contain conflicting ACL snapshots")
        policies[row["uri"]] = policy
        slot = (row["uri"], row.get("level"))
        if slot in slots and slots[slot] != fields:
            raise FailedPreconditionError("Document controls contain conflicting snapshots")
        slots[slot] = fields


async def _target_records(fs, uri, inventory, old, root_is_file, ctx):
    shapes = {
        "": not root_is_file,
        **{key: value.is_dir for key, value in inventory.entries.items()},
    }
    _validate_acl_snapshots(old.records)
    old_by_slot = {(row["uri"][len(uri) :], row.get("level")): row for row in old.records}
    old_root = [row for row in old.records if row["uri"] == uri]
    if any("#" in row["uri"][len(uri) :] for row in old.records):
        raise FailedPreconditionError("Fragment-control replacement is unsupported")
    cross = old.node["kind"] == "file" and not root_is_file
    if cross:
        if any(row.get("level") != 2 for row in old_root):
            raise FailedPreconditionError("Document root controls conflict")
        controls = [_controls_for_level(row) for row in old_root]
        if controls and any(row != controls[0] for row in controls):
            raise FailedPreconditionError("Document root controls conflict")
    records = []
    slots = set()
    for rel, is_dir in sorted(shapes.items()):
        member = uri + "/" + rel if rel else uri
        _uri(fs, member, ctx)
        for level in (0, 1) if is_dir else (2,):
            slots.add(("/" + rel if rel else "", level))
            record = _root_record(member, level, ctx, is_dir)
            previous = old_by_slot.get(("/" + rel if rel else "", level))
            if previous is not None:
                record.update(tree_replace._controls(previous))
            elif cross and not rel and old_root:
                record.update(controls[0])
            records.append(record)
    for row in old.records:
        suffix = row["uri"][len(uri) :]
        if (
            suffix in {"/" + rel if rel else "" for rel in shapes}
            and (suffix, row.get("level")) not in slots
            and not (cross and not suffix)
        ):
            if any(
                value not in (None, "", [], {}, False, 0)
                for key, value in tree_replace._controls(row).items()
                if key in {"acl_direct_grants", "search_tags", "tags"}
                or key == "acl_mode"
                and value == "restricted"
            ):
                raise FailedPreconditionError("Retained document controls lack an output slot")
    if len(records) > RECORD_LIMIT:
        raise ResourceExhaustedError("Document context limit exceeded")
    if fs.acl_manager is None:
        raise FailedPreconditionError("Document ACL manager unavailable")
    _, records = await fs.acl_manager.prepare_subtree_records(uri, records, ctx)
    backend = await fs._get_vector_store()._get_backend_for_context(ctx)
    return await backend._async_adapter.run(backend._prepare_upsert_payloads, records)


async def _preflight_effects(fs, uri, inventory, old, records, root_is_file, ctx):
    ancestors = []
    parent = uri.rsplit("/", 1)[0]
    while parent.startswith("viking://resources"):
        ancestors.append(parent)
        if parent == "viking://resources":
            break
        parent = parent.rsplit("/", 1)[0]
    directories = (
        0 if root_is_file else 1 + sum(entry.is_dir for entry in inventory.entries.values())
    )
    new_nodes = len(inventory.entries) + (0 if root_is_file else 1) + 2 * directories
    old_nodes = len(old.entries) + (1 if old.node["kind"] == "directory" else 0)
    if new_nodes + old_nodes + 2 * len(ancestors) > FORMAL_LIMIT:
        raise ResourceExhaustedError("Document operation formal-node limit exceeded")
    backup_anchor = int(
        old.node["kind"] != "absent" and not any(row["uri"] == uri for row in old.records)
    )
    if len(old.records) + len(records) + backup_anchor > RECORD_LIMIT:
        raise ResourceExhaustedError("Document operation context limit exceeded")
    metadata_roots = set(ancestors) | {row["uri"] for row in records if row["level"] in {0, 1}}
    effects = [*ancestors]
    for root in metadata_roots:
        effects.extend([root + "/.abstract.md", root + "/.overview.md"])
    await fs._ensure_access_many(effects, ctx, action=AclAction.WRITE)


async def _publish_records(fs, uri, records, ctx):
    store = fs._get_vector_store()
    backend = await store._get_backend_for_context(ctx)
    records = await backend._async_adapter.run(backend._prepare_upsert_payloads, records)
    if records:
        ids = await store._upsert_many_raw(records, ctx=ctx)
        if len(ids) != len(records):
            raise RuntimeError("Document policy publication incomplete")
    actual = await _records(fs, uri, ctx)
    by_id = {row["id"]: row for row in actual}
    for record in records:
        found = by_id.get(record["id"])
        if found is None or any(
            tree_replace._controls(found).get(key) != value
            for key, value in tree_replace._controls(record).items()
        ):
            raise RuntimeError("Document policy verification failed")
    return [by_id[row["id"]] for row in records]


async def _prepare(service, request, private, ctx):
    store = LocalParseOutputStore(str(private / "artifacts"))
    name = request.source_name
    if Path(name).suffix.lower() not in {
        ".md",
        ".markdown",
        ".mdown",
        ".mkd",
        ".txt",
        ".text",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".rst",
    }:
        raise InvalidArgumentError("Literal document parser is unsupported for this source_name")
    source = private / name
    source.write_bytes(request.text.encode())
    local = LocalResource(
        path=source, source_type=SourceType.LOCAL, original_source=name, is_temporary=False
    )
    processor = await service.resources._resource_processor._media_processor_for(ctx)
    parsed = await processor.process(
        source=str(source),
        prepared_resource=local,
        source_name=name,
        instruction=request.reason,
        parse_output_store=store,
        parser_backend="internal",
        parse_mode="no_split" if request.representation == "raw_file" else "default",
        allow_media_resolution=False,
    )
    ref = parsed.artifact_ref
    resolved = await resolve_artifact_doc_root(
        store, ref, flatten_single_file=request.representation == "raw_file"
    )
    if request.representation == "raw_file" and not resolved.root_is_file:
        raise FailedPreconditionError("Raw document parser did not produce one file")
    if request.representation in {"verbatim", "exact_directory"}:
        await store.cleanup(ref)
        writer = await ParseArtifactWriter.create(store)
        await writer.write_bytes("0001/0000.md", request.text.encode())
        ref = await writer.finalize()
        doc_rel, root_is_file = "", False
    else:
        doc_rel, root_is_file = resolved.doc_rel, resolved.root_is_file
    await service.resources._resource_processor._sanitize_parse_artifact(
        store=store,
        artifact_ref=ref,
        doc_rel=doc_rel,
        target_root_uri=request.target_uri,
        ctx=ctx,
        root_is_file=root_is_file,
    )
    inventory = await prepare_artifact_inventory(
        store, ref, doc_rel=doc_rel, target_root_uri=request.target_uri, root_is_file=root_is_file
    )
    directories = (
        0 if root_is_file else 1 + sum(value.is_dir for value in inventory.entries.values())
    )
    if len(inventory.entries) + (0 if root_is_file else 1) + 2 * directories > FORMAL_LIMIT:
        raise ResourceExhaustedError("Document formal-node limit exceeded")
    if request.representation in {"raw_file", "verbatim", "exact_directory"}:
        rel = doc_rel if root_is_file else "0001/0000.md"
        if await store.read_bytes(ref, rel) != request.text.encode():
            raise FailedPreconditionError("Exact document input was transformed")
    return store, ref, doc_rel, root_is_file, inventory


async def _embed_private_records(fs, records, root_uri, bodies, ctx, admitted, lease):
    """Use the configured embedder before any formal policy/body publication."""
    for record in records:
        if record.get("vector") is not None:
            continue
        relative = record["uri"][len(root_uri) :].lstrip("/")
        if record["level"] == 2:
            raw = bodies[relative]
        else:
            raw = b"\n".join(
                value
                for path, value in sorted(bodies.items())
                if not relative or path == relative or path.startswith(relative + "/")
            )
        if len(raw) > MAX_TEXT_BYTES:
            raise ResourceExhaustedError("Private policy embedding input limit exceeded")
        await _live(fs, ctx, admitted, lease)
        embedded = await fs._get_embedder(ctx).embed_async(
            record["uri"] + "\n" + raw.decode("utf-8", errors="replace"),
            is_query=False,
        )
        await _live(fs, ctx, admitted, lease)
        if not embedded.dense_vector:
            raise FailedPreconditionError("Private policy embedding failed")
        record["vector"] = embedded.dense_vector


async def _copy_snapshot(fs, source, destination, old, records, ctx, lease):
    entries, bodies = old.entries, old.bodies
    if old.node["kind"] == "directory" and source != destination:
        entries = await tree_replace._expected_transfer_entries(
            fs,
            source,
            destination,
            old.entries,
            fs._pathlock_fs_ctx(ctx, lease),
            source_path=fs._uri_to_path(source, ctx=ctx),
        )
        bodies = dict(old.bodies)
        for rel, raw in old.bodies.items():
            parent, _, name = rel.rpartition("/")
            if name in ABSTRACT_OVERVIEW_FILENAMES:
                bodies[rel] = rewrite_abstract_overview_for_transfer(
                    raw,
                    level=ContextLevel.ABSTRACT
                    if name == ".abstract.md"
                    else ContextLevel.OVERVIEW,
                    source_dir_uri=source + "/" + parent if parent else source,
                    target_dir_uri=destination + "/" + parent if parent else destination,
                    source_scope_uri=source,
                    target_scope_uri=destination,
                ).encode("utf-8")
    published = await _publish_records(fs, destination, records, ctx)
    if old.node["kind"] == "directory":
        await fs._mkdir_transfer_root_exclusive(
            fs._uri_to_path(destination, ctx=ctx), destination, ctx, lease
        )
        for entry in old.entries:
            if entry["type"] == "dir":
                await fs.mkdir(destination + "/" + entry["path"], ctx=ctx, lease_ref=lease)
    for rel, raw in bodies.items():
        target = destination + "/" + rel if rel else destination
        await fs.write_file_bytes(target, raw, ctx=ctx, lease_ref=lease)
    expected = _descriptor(destination, entries, published, bodies, old.node["kind"])
    await tree_replace._finish_context_copy(fs, destination, records, ctx)
    actual = await _snapshot(fs, destination, ctx, lease)
    if actual.node != expected:
        raise RuntimeError("Document backup verification failed")
    return actual


async def _clear_body(fs, uri, old, ctx, lease):
    if old.node["kind"] != "absent":
        await fs._async_agfs.rm(
            fs._uri_to_path(uri, ctx=ctx),
            recursive=old.node["kind"] == "directory",
            fs_ctx=fs._pathlock_fs_ctx(ctx, lease),
        )


async def write_document(service, request, ctx, *, fault=None):
    """Own admission, private artifact, verified backup and final queue handoff."""

    async def run():
        return await _write_document(service, request, ctx, fault=fault)

    return await service.fs._finish_transfer_after_caller_cancel(run(), operation="document-write")


async def _write_document(service, request, ctx, *, fault):
    fs, uri = service.viking_fs, request.target_uri
    _uri(fs, uri, ctx)
    uris = [uri]
    if request.backup_uri is not None:
        _uri(fs, request.backup_uri, ctx)
        fs._validate_transfer_paths(
            fs._uri_to_path(uri, ctx=ctx), fs._uri_to_path(request.backup_uri, ctx=ctx)
        )
        uris.append(request.backup_uri)
    for member in uris:
        await fs._ensure_access(member, ctx, action=AclAction.WRITE)
        await fs._ensure_access(member, ctx, action=AclAction.MANAGE)
    admitted = await fs._capture_resource_identity(ctx)
    lease = await fs._acquire_resource_coverage(ctx, admitted)
    original = None
    backup = None
    backup_started = False
    recovery_records = []
    mutated = False
    committed = False

    async def step(phase):
        await _live(fs, ctx, admitted, lease)
        if fault is not None:
            await fault(phase)
        await _live(fs, ctx, admitted, lease)

    try:
        async with service.fs._uri_mutation_coordinator.mutation(ctx.account_id, uris):
            await service.fs._refuse_watched_resource_targets(ctx, uris)
            for member in uris:
                await fs._ensure_transfer_parent_directory(
                    fs._uri_to_path(member, ctx=ctx), member, operation="document-write"
                )
            original = await _snapshot(fs, uri, ctx, lease)
            if original.node != request.expected.model_dump():
                raise ConflictError("Document expectation changed")
            await fs._ensure_access_many(
                [
                    uri,
                    *(row["uri"] for row in original.records),
                    *(uri + "/" + row["path"] for row in original.entries if row["path"]),
                ],
                ctx,
                action=AclAction.MANAGE,
            )
            if (
                request.backup_uri is not None
                and (await _snapshot(fs, request.backup_uri, ctx, lease)).node["kind"] != "absent"
            ):
                raise ConflictError("Document backup destination is occupied")
            _validate_acl_snapshots(original.records)
            config_root = Path(get_openviking_config().storage.parse_output.resolved_local_root())
            config_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="document-write-", dir=config_root
            ) as directory:
                private = Path(directory)
                store, ref, doc_rel, root_is_file, inventory = await _prepare(
                    service, request, private, ctx
                )
                records = await _target_records(fs, uri, inventory, original, root_is_file, ctx)
                await _preflight_effects(fs, uri, inventory, original, records, root_is_file, ctx)
                target = AgfsResourceTarget(viking_fs=fs, root_uri=uri, ctx=ctx, lease_ref=lease)
                intent = _DocumentIntent.from_ingest_options(
                    target_uri=uri,
                    processing_mode="semantic_and_vectors",
                    ingest_options=IngestOptions(),
                    vectorize=True,
                )
                snapshot = await build_rnfv_snapshot(
                    viking_fs=fs,
                    vikingdb=service.resources._resource_processor.vikingdb,
                    store=store,
                    artifact_ref=ref,
                    target_uri=uri,
                    ctx=ctx,
                    doc_rel=doc_rel,
                    request_intent=intent,
                    root_is_file=root_is_file,
                    formal_snapshot=({}, True),
                    artifact_inventory=inventory,
                    vector_inventory={row["id"]: row for row in records},
                )
                _, plan = await build_context_update_plan_from_snapshot(
                    snapshot=snapshot,
                    store=_DocRelStore(store, doc_rel),
                    artifact_ref=ref,
                    target=target,
                    vikingdb=service.resources._resource_processor.vikingdb,
                    context_type="resource",
                    is_code_repo=False,
                    ctx=ctx,
                    account_id=ctx.account_id,
                    root_preexisting=False,
                    artifact_paths=inventory.artifact_paths,
                    ingest_options=IngestOptions(),
                    root_is_file=root_is_file,
                )
                planned = len(plan.direct_index_actions)
                if plan.semantic_plan is not None:
                    planned += sum(
                        len(entry.index_slots) for entry in plan.semantic_plan.tree.entries
                    )
                if planned > ACTION_LIMIT:
                    raise ResourceExhaustedError("Document index-action limit exceeded")
                artifact_bodies = {
                    rel: await store.read_bytes(ref, path)
                    for rel, path in inventory.artifact_paths.items()
                    if not inventory.entries[rel].is_dir
                }
                if sum(map(len, artifact_bodies.values())) > MAX_TEXT_BYTES:
                    raise ResourceExhaustedError("Private artifact embedding limit exceeded")
                if (
                    original.node["kind"] != "absent"
                    and not any(row["uri"] == uri for row in original.records)
                    and sum(map(len, original.bodies.values())) > MAX_TEXT_BYTES
                ):
                    raise ResourceExhaustedError("Unindexed backup embedding limit exceeded")
                await _embed_private_records(
                    fs, records, uri, artifact_bodies, ctx, admitted, lease
                )
                await step("prepared")
                if (await _snapshot(fs, uri, ctx, lease)).node != original.node:
                    raise ConflictError("Document expectation changed")
                if request.backup_uri is not None:
                    old_records = original.records
                    if not any(row["uri"] == uri for row in old_records):
                        old_records = [
                            *old_records,
                            _root_record(
                                uri,
                                2 if original.node["kind"] == "file" else 0,
                                ctx,
                                original.node["kind"] == "directory",
                            ),
                        ]
                    anchor = await fs.acl_manager.resolve(uri, ctx)
                    recovery_records = deepcopy(old_records)
                    if not any(row["uri"] == uri for row in original.records):
                        _, recovery_records = await fs.acl_manager.prepare_subtree_records(
                            uri,
                            recovery_records,
                            ctx,
                            root_direct=anchor.permissions,
                            root_mode=AclMode.RESTRICTED,
                        )
                    await _embed_private_records(
                        fs, recovery_records, uri, original.bodies, ctx, admitted, lease
                    )
                    backup_records = await tree_replace._prepare_context_copy(
                        fs, uri, request.backup_uri, old_records, ctx, anchor=anchor
                    )
                    _, backup_records = await fs.acl_manager.prepare_subtree_records(
                        request.backup_uri,
                        backup_records,
                        ctx,
                        root_direct=anchor.permissions,
                        root_mode=AclMode.RESTRICTED,
                    )
                    await _embed_private_records(
                        fs,
                        backup_records,
                        request.backup_uri,
                        original.bodies,
                        ctx,
                        admitted,
                        lease,
                    )
                    await step("backup")
                    if (await _snapshot(fs, uri, ctx, lease)).node != original.node:
                        raise ConflictError("Document expectation changed")
                    if (await _snapshot(fs, request.backup_uri, ctx, lease)).node[
                        "kind"
                    ] != "absent":
                        raise ConflictError("Document backup destination is occupied")
                    backup_started = True
                    backup = await _copy_snapshot(
                        fs, uri, request.backup_uri, original, backup_records, ctx, lease
                    )
                await step("publish")
                # Keep all old restrictive records until old bytes have gone.
                mutated = True
                await _publish_records(fs, uri, records, ctx)
                await _clear_body(fs, uri, original, ctx, lease)
                await step("install")
                await execute_content_tree_actions(
                    plan.content_tree_actions,
                    store=_DocRelStore(store, doc_rel),
                    artifact_ref=ref,
                    target=target,
                )
                await tree_replace._finish_context_copy(fs, uri, records, ctx)
                installed = await _snapshot(fs, uri, ctx, lease)
                expected_bodies = {
                    rel: await store.read_bytes(ref, path)
                    for rel, path in inventory.artifact_paths.items()
                    if not inventory.entries[rel].is_dir
                }
                if installed.bodies != expected_bodies or installed.node["kind"] != (
                    "file" if root_is_file else "directory"
                ):
                    raise RuntimeError("Document installed bytes verification failed")
                # Read policy back before ACK and before handing off the original owner.
                expected_controls = {row["id"]: tree_replace._controls(row) for row in records}
                if set(expected_controls) != {row["id"] for row in installed.records}:
                    raise RuntimeError("Document installed policy inventory changed")
                for row in installed.records:
                    expected_control = expected_controls.get(row["id"])
                    if expected_control is None or any(
                        tree_replace._controls(row).get(key) != value
                        for key, value in expected_control.items()
                    ):
                        raise RuntimeError("Document installed policy verification failed")
                await step("verified")
                committed = True
                result = {
                    "uri": uri,
                    "node": installed.node,
                    "input_sha256": hashlib.sha256(request.text.encode()).hexdigest(),
                    "representation": "raw_file"
                    if root_is_file
                    else "exact_directory"
                    if request.representation == "exact_directory"
                    else "verbatim_directory"
                    if request.representation == "verbatim"
                    else "parsed_directory",
                    "content_state": "committed",
                    "processing_state": "unobserved",
                    "backup": {"uri": request.backup_uri, "node": backup.node} if backup else None,
                    "cleanup_state": "complete",
                }
                try:
                    conditions = await capture_resource_index_conditions(
                        plan.direct_index_actions, ctx, admitted, lease
                    )
                    prepared = {
                        "root_uri": uri,
                        "temp_uri": uri,
                        "temp_dir_path": None,
                        "artifact_ref": None,
                        "source_committed": True,
                        "resources_only_document": True,
                        "resource_creation_identity": admitted.creation_identity,
                        "resource_index_conditions": conditions,
                        "target_preexisting": original.node["kind"] != "absent",
                        "root_is_file": root_is_file,
                        "is_code_repo": False,
                        "incremental_noop": False,
                        "context_update_plan": plan.after_content_commit().to_dict(),
                        "semantic_source": None,
                    }
                    handoff = await service.resources._lock_to_handoff_payload(lease)
                    msg = AddResourceMsg(
                        task_id=str(uuid4()),
                        job_phase=AddResourcePhase.POST_PROCESS,
                        resource_creation_identity=admitted.creation_identity,
                        root_uri=uri,
                        prepared=prepared,
                        account_id=ctx.account_id,
                        user_id=ctx.user.user_id,
                        role=str(ctx.role),
                        group_ids=list(ctx.group_ids),
                        actor_peer_id=ctx.actor_peer_id,
                        bypass_acl=ctx.bypass_acl,
                        lock_handoff=handoff,
                        reason=request.reason,
                        instruction=request.reason,
                        source_name=request.source_name,
                        build_index=True,
                        summarize=False,
                        processing_mode="semantic_and_vectors",
                        skip_watch_management=True,
                        directly_upload_media=False,
                    )
                    enqueue_lease, lease = lease, None
                    await service.resources._enqueue_add_resource_job(
                        msg, queue_name=QueueManager.ADD_RESOURCE, resource_lock=enqueue_lease
                    )
                    result["processing_state"] = "accepted"
                except Exception:
                    result["processing_state"] = "unobserved"
                return result
    except Exception as failure:
        if committed:
            raise OpenVikingError(
                "Document committed; later outcome unavailable",
                code="INTERNAL",
                details={
                    "uri": uri,
                    "content_state": "committed",
                    "processing_state": "unobserved",
                },
            ) from None
        state = "not_started"
        if mutated:
            state = "rollback_unavailable"
            try:
                await _live(fs, ctx, admitted, lease)
                current = await _snapshot(fs, uri, ctx, lease, owned_absence=True)
                await _clear_body(fs, uri, current, ctx, lease)
                if backup is not None:
                    if (await _snapshot(fs, request.backup_uri, ctx, lease)).node != backup.node:
                        raise RuntimeError("Document backup changed")
                    await _copy_snapshot(fs, uri, uri, original, recovery_records, ctx, lease)
                    await tree_replace._finish_context_copy(fs, uri, original.records, ctx)
                else:
                    await tree_replace._finish_context_copy(fs, uri, [], ctx)
                restored = await _snapshot(fs, uri, ctx, lease)
                if restored.node != original.node:
                    raise RuntimeError("Document rollback verification failed")
                state = "rolled_back"
            except Exception:
                state = "rollback_failed"
        details = {"uri": uri, "content_state": state, "processing_state": "unobserved"}
        if backup_started:
            details["retained_copies"] = [
                {"uri": request.backup_uri, "node": backup.node if backup else None}
            ]
        if not mutated and isinstance(failure, OpenVikingError):
            failure.message = "Document precondition failed"
            failure.args = (failure.message,)
            failure.details = details
            raise
        raise OpenVikingError(
            "Document publication failed", code="INTERNAL", details=details
        ) from failure
    finally:
        if lease is not None:
            with suppress(Exception):
                await fs._async_agfs.pathlock_release(lease)
