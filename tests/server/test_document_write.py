"""Native document-write vertical slices. Only model providers are synthetic."""

import asyncio

import pytest

from openviking.config.embedding import AccountBoundEmbedder
from openviking.server.auth import get_request_context
from openviking.server.identity import RequestContext, Role
from openviking.service.document_write import DocumentWriteRequest, document_node, write_document
from openviking.storage.acl import AclMode
from openviking.utils.media_processor import UnifiedResourceProcessor
from openviking_cli.exceptions import ConflictError, OpenVikingError, PermissionDeniedError
from openviking_cli.session.user_id import UserIdentifier
from tests.server import conftest as server_fixtures
from tests.server import test_real_resource_producer_e2e as producer_fixtures

app = server_fixtures.app
client = server_fixtures.client
service = server_fixtures.service
temp_dir = server_fixtures.temp_dir
admitted_service = producer_fixtures.admitted_service
_root_ctx = producer_fixtures._root_ctx

BASE = "viking://resources"
TARGET = BASE + "/document.md"
BACKUP = BASE + "/saved.md"
HINT = "Process the literal cobalt decision."


@pytest.fixture(autouse=True)
def normal_account_embedder(service, monkeypatch):
    """Exercise the supported account API, not the shared fixture legacy shortcut."""
    monkeypatch.setattr(service.viking_fs, "query_embedder", None)
    assert service.viking_fs._embedding_provider is not None


async def settle(service):
    await asyncio.wait_for(service.resources.wait_processed(), timeout=35)


async def seed_file(service, text=b"original\n", restricted=False):
    ctx = _root_ctx()
    fs = service.viking_fs
    await fs.write_file_bytes(TARGET, text, ctx=ctx)
    await fs._get_vector_store().upsert(
        {
            "id": TARGET,
            "uri": TARGET,
            "level": 2,
            "account_id": ctx.account_id,
            "abstract": "original",
            "vector": [0.1] * 2048,
            "search_tags": ["owner=retained"],
        },
        ctx=ctx,
    )
    if restricted:
        await service.runtime_config_manager.patch_account(
            ctx.account_id, {"acl": {"enabled": True}}
        )
        await fs.set_acl(
            TARGET,
            [{"principal": "user:" + ctx.user.user_id, "level": "manage"}],
            acl_mode=AclMode.RESTRICTED,
            ctx=ctx,
        )
    return (await document_node(service, TARGET, ctx))["node"]


async def test_http_private_parser_and_contract(admitted_service, app, client, monkeypatch):
    service = admitted_service
    ctx = _root_ctx()
    app.dependency_overrides[get_request_context] = lambda: ctx
    seen = []
    excluded_links = []

    async def excluded_link(*args, **kwargs):
        excluded_links.append(True)
        raise AssertionError("resources-only writes must not link excluded memory")

    monkeypatch.setattr(service.resources, "_link_resource_reason_memory", excluded_link)
    original = UnifiedResourceProcessor.process

    async def process(self, *args, **kwargs):
        seen.append(kwargs["instruction"])
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(UnifiedResourceProcessor, "process", process)
    from openviking.parse.parsers.markdown import MarkdownParser

    async def forbidden(*args, **kwargs):
        raise AssertionError("literal text must not resolve image files")

    monkeypatch.setattr(MarkdownParser, "_ingest_local_images", forbidden)
    contract = await client.get("/api/v1/resources/document-write-contract")
    assert contract.json()["result"] == {"contract": "resources-document-write", "version": 1}
    before = await client.get("/api/v1/resources/document-node", params={"uri": TARGET})
    assert before.json()["result"]["node"] == {"kind": "absent"}
    reply = await asyncio.wait_for(
        client.post(
            "/api/v1/resources/document-write",
            json={
                "target_uri": TARGET,
                "source_name": "note.md",
                "text": "# Note\n\n[[Other]] ![x](/no-access/private.png)\n",
                "reason": HINT,
                "expected": {"kind": "absent"},
            },
        ),
        timeout=45,
    )
    assert reply.status_code == 200, reply.text
    result = reply.json()["result"]
    assert result["content_state"] == "committed"
    assert result["node"]["kind"] == "directory"
    assert result["processing_state"] == "accepted", result
    assert seen == [HINT]
    await settle(service)
    records, _ = await service.viking_fs._get_vector_store().scroll(limit=200, ctx=ctx)
    assert any(
        row.get("uri", "").startswith(TARGET)
        and row.get("level") == 2
        and row.get("abstract")
        and row.get("md5")
        for row in records
    )

    observed = (await document_node(service, TARGET, ctx))["node"]
    changed = dict(observed, tree_revision="tree-sha256:" + "0" * 64)
    with pytest.raises(ConflictError):
        await write_document(
            service,
            DocumentWriteRequest(
                target_uri=TARGET,
                source_name="replacement.md",
                text="replacement",
                expected=changed,
                backup_uri=BACKUP,
            ),
            ctx,
        )
    replaced = await write_document(
        service,
        DocumentWriteRequest(
            target_uri=TARGET,
            source_name="replacement.md",
            text="# Replacement\n\nNew directory document.\n",
            expected=observed,
            backup_uri=BACKUP,
        ),
        ctx,
    )
    assert replaced["node"]["kind"] == "directory"
    assert replaced["backup"]["node"]["kind"] == "directory"
    assert replaced["processing_state"] == "accepted"
    await settle(service)
    assert excluded_links == []


async def test_raw_preservation_and_protected_backup(admitted_service, monkeypatch):
    service = admitted_service
    ctx = _root_ctx()
    expected = await seed_file(service, restricted=True)
    outsider = RequestContext(user=UserIdentifier(ctx.account_id, "outsider"), role=Role(Role.USER))
    fs = service.viking_fs
    original = fs.write_file_bytes
    observed = []

    async def write(uri, data, *args, **kwargs):
        result = await original(uri, data, *args, **kwargs)
        if uri in {TARGET, BACKUP}:
            with pytest.raises(PermissionDeniedError):
                await fs.read_file_bytes(uri, ctx=outsider)
            observed.append(uri)
        return result

    monkeypatch.setattr(fs, "write_file_bytes", write)
    body = " \nnew exact raw\r\n "
    result = await write_document(
        service,
        DocumentWriteRequest(
            target_uri=TARGET,
            source_name="note.txt",
            text=body,
            reason=HINT,
            expected=expected,
            representation="raw_file",
            backup_uri=BACKUP,
        ),
        ctx,
    )
    assert result["content_state"] == "committed", result
    assert result["node"]["kind"] == "file"
    assert result["backup"]["node"]["content_sha256"] == expected["content_sha256"]
    assert await fs.read_file_bytes(TARGET, ctx=ctx) == body.encode()
    assert await fs.read_file_bytes(BACKUP, ctx=ctx) == b"original\n"
    assert set(observed) == {TARGET, BACKUP}
    await settle(service)


async def test_file_to_verbatim_and_root_controls(admitted_service):
    service = admitted_service
    ctx = _root_ctx()
    expected = await seed_file(service, restricted=True)
    target = BASE + "/document.verbatim"
    # Seed this transition at the exact composed verbatim URI.
    fs = service.viking_fs
    await fs.write_file_bytes(target, b"old", ctx=ctx)
    await fs._get_vector_store().upsert(
        {
            "id": target,
            "uri": target,
            "level": 2,
            "account_id": ctx.account_id,
            "abstract": "old",
            "vector": [0.1] * 2048,
            "search_tags": ["owner=old"],
        },
        ctx=ctx,
    )
    await fs.set_acl(
        target,
        [{"principal": "user:" + ctx.user.user_id, "level": "manage"}],
        acl_mode=AclMode.RESTRICTED,
        ctx=ctx,
    )
    expected = (await document_node(service, target, ctx))["node"]
    body = "\n  exact\r\nUnicode Ω\n "
    result = await write_document(
        service,
        DocumentWriteRequest(
            target_uri=target,
            source_name="note.md",
            text=body,
            reason=HINT,
            representation="verbatim",
            expected=expected,
            backup_uri=BACKUP,
        ),
        ctx,
    )
    assert result["node"]["kind"] == "directory", result
    assert await fs.read_file_bytes(target + "/0001/0000.md", ctx=ctx) == body.encode()
    await settle(service)
    records, _ = await fs._get_vector_store().scroll(limit=200, ctx=ctx)
    roots = [row for row in records if row.get("uri") == target and row.get("level") in {0, 1}]
    assert roots and all(row["search_tags"] == ["owner=old"] for row in roots)


async def test_stale_controls_and_same_process_rollback(admitted_service, monkeypatch):
    service = admitted_service
    ctx = _root_ctx()
    expected = await seed_file(service)
    request = DocumentWriteRequest(
        target_uri=TARGET,
        source_name="note.txt",
        text="new",
        representation="raw_file",
        expected=expected,
        backup_uri=BACKUP,
    )
    changed = dict(expected, control_revision="file-controls-v1-sha256:" + "0" * 64)
    with pytest.raises(ConflictError):
        await write_document(
            service, request.model_copy(update={"expected": type(request.expected)(**changed)}), ctx
        )
    assert (await document_node(service, BACKUP, ctx))["node"]["kind"] == "absent"

    async def fail_embedding(*args, **kwargs):
        raise RuntimeError("injected private embedding failure")

    with monkeypatch.context() as patch:
        patch.setattr(AccountBoundEmbedder, "embed_async", fail_embedding)
        with pytest.raises(OpenVikingError) as embedding_error:
            await write_document(service, request, ctx)
        assert embedding_error.value.details["content_state"] == "not_started"
        assert (await document_node(service, TARGET, ctx))["node"] == expected
        assert (await document_node(service, BACKUP, ctx))["node"]["kind"] == "absent"

    phases = []

    async def fault(phase):
        phases.append(phase)
        if phase == "install":
            raise OSError("injected install failure")

    with pytest.raises(OpenVikingError) as raised:
        await write_document(service, request, ctx, fault=fault)
    assert "install" in phases
    assert raised.value.details["content_state"] == "rolled_back", raised.value.details
    assert raised.value.details["retained_copies"][0]["uri"] == BACKUP
    assert (await document_node(service, TARGET, ctx))["node"] == expected


async def test_conflicting_acl_refuses_before_private_preparation(admitted_service, monkeypatch):
    from openviking.service import document_write
    from openviking.storage.vector_ids import vector_record_id
    from openviking_cli.exceptions import FailedPreconditionError

    service = admitted_service
    fs = service.viking_fs
    ctx = _root_ctx()
    await fs.mkdir(TARGET, ctx=ctx)
    await fs.write_file_bytes(TARGET + "/original.md", b"original body", ctx=ctx)
    rows = []
    for level in (0, 1):
        rows.append(
            {
                "id": vector_record_id(ctx.account_id, TARGET, level),
                "uri": TARGET,
                "level": level,
                "account_id": ctx.account_id,
                "abstract": "original",
                "vector": [0.1] * 2048,
                "acl_mode": "restricted" if level == 0 else "inherit",
                "acl_direct_grants": ["7:user:" + ctx.user.user_id] if level == 0 else [],
                "acl_inherited_grants": ["7:user:*"],
            }
        )
    await fs._get_vector_store()._upsert_many_raw(rows, ctx=ctx)
    before = (await document_node(service, TARGET, ctx))["node"]
    request = DocumentWriteRequest(
        target_uri=TARGET,
        source_name="replacement.md",
        text="replacement",
        expected=before,
        backup_uri=BACKUP,
    )
    assert request.expected.kind == "directory"
    preparation_calls = []
    publication_calls = []
    validated_slots = []
    original_validation = document_write._validate_acl_snapshots

    def validate(rows):
        validated_slots.append({(row["uri"], row["level"]) for row in rows})
        return original_validation(rows)

    monkeypatch.setattr(document_write, "_validate_acl_snapshots", validate)

    async def forbid_prepare(*args, **kwargs):
        preparation_calls.append(True)
        raise AssertionError("conflicting ACLs must refuse before private preparation")

    async def forbid_publication(*args, **kwargs):
        publication_calls.append(True)
        raise AssertionError("conflicting ACLs must refuse before publication")

    monkeypatch.setattr(document_write, "_prepare", forbid_prepare)
    monkeypatch.setattr(document_write, "_publish_records", forbid_publication)
    monkeypatch.setattr(fs, "write_file_bytes", forbid_publication)
    with pytest.raises(FailedPreconditionError) as refused:
        await write_document(service, request, ctx)
    assert refused.value.details["content_state"] == "not_started"
    assert validated_slots == [{(TARGET, 0), (TARGET, 1)}]
    assert preparation_calls == []
    assert publication_calls == []
    assert (await document_node(service, TARGET, ctx))["node"] == before
    assert await fs.read_file_bytes(TARGET + "/original.md", ctx=ctx) == b"original body"
    assert (await document_node(service, BACKUP, ctx))["node"] == {"kind": "absent"}


@pytest.mark.parametrize("kind", ["file", "directory"])
async def test_unindexed_original_rollback(admitted_service, monkeypatch, kind):
    from openviking.service import document_write

    service, ctx = admitted_service, _root_ctx()
    fs = service.viking_fs
    parent, target = BASE + "/private", BASE + "/private/original"
    await fs.mkdir(parent, ctx=ctx)
    row = document_write._root_record(parent, 0, ctx, True)
    row["vector"] = [0.1] * 2048
    await fs._get_vector_store().upsert(row, ctx=ctx)
    await service.runtime_config_manager.patch_account(ctx.account_id, {"acl": {"enabled": True}})
    await fs.set_acl(
        parent,
        [{"principal": "user:" + ctx.user.user_id, "level": "manage"}],
        acl_mode=AclMode.RESTRICTED,
        ctx=ctx,
    )
    if kind == "directory":
        await fs.mkdir(target, ctx=ctx)
    member = target + "/original.md" if kind == "directory" else target
    await fs.write_file_bytes(member, b"original exact\r\n ", ctx=ctx)
    expected = (await document_node(service, target, ctx))["node"]
    assert await document_write._records(fs, target, ctx) == []
    outsider = RequestContext(user=UserIdentifier(ctx.account_id, "outsider"), role=Role(Role.USER))
    original_write = fs.write_file_bytes
    first_bytes, phases, embedded = [], [], []
    original_embed = AccountBoundEmbedder.embed_async

    async def embed(self, *args, **kwargs):
        assert "install" not in phases, "rollback must not call the model"
        embedded.append(args[0])
        return await original_embed(self, *args, **kwargs)

    async def protected_write(uri, data, *args, **kwargs):
        result = await original_write(uri, data, *args, **kwargs)
        if uri == member or uri == BACKUP or uri.startswith(BACKUP + "/"):
            with pytest.raises(PermissionDeniedError):
                await fs.read_file_bytes(uri, ctx=outsider)
            first_bytes.append(uri)
        return result

    async def fault(phase):
        phases.append(phase)
        if phase == "install":
            raise OSError("one-shot install fault")

    monkeypatch.setattr(AccountBoundEmbedder, "embed_async", embed)
    monkeypatch.setattr(fs, "write_file_bytes", protected_write)
    with pytest.raises(OpenVikingError) as raised:
        await write_document(
            service,
            DocumentWriteRequest(
                target_uri=target,
                source_name="replacement.md",
                text="new body",
                expected=expected,
                backup_uri=BACKUP,
                representation="raw_file" if kind == "file" else "ingest",
            ),
            ctx,
            fault=fault,
        )
    assert "install" in phases
    assert raised.value.details["content_state"] == "rolled_back", raised.value.details
    assert (await document_node(service, target, ctx))["node"] == expected
    assert await fs.read_file_bytes(member, ctx=ctx) == b"original exact\r\n "
    backup_member = BACKUP + "/original.md" if kind == "directory" else BACKUP
    assert await fs.read_file_bytes(backup_member, ctx=ctx) == b"original exact\r\n "
    assert (await document_node(service, BACKUP, ctx))["node"] == raised.value.details[
        "retained_copies"
    ][0]["node"]
    assert await document_write._records(fs, target, ctx) == []
    assert member in first_bytes and backup_member in first_bytes
    assert any(value.startswith(target + "\n") for value in embedded)


@pytest.mark.parametrize("input_path", ["context", "physical"])
async def test_document_node_hides_denied_descendant(
    admitted_service, app, client, monkeypatch, input_path
):
    from openviking.service import document_write

    service, ctx = admitted_service, _root_ctx()
    fs = service.viking_fs
    hidden = TARGET + "/classified-child.md"
    await fs.mkdir(TARGET, ctx=ctx)
    await fs.write_file_bytes(hidden, b"secret", ctx=ctx)
    await service.runtime_config_manager.patch_account(ctx.account_id, {"acl": {"enabled": True}})
    row = document_write._root_record(hidden, 2, ctx, False)
    row.update(
        vector=[0.1] * 2048,
        acl_mode="restricted",
        acl_direct_grants=["7:user:" + ctx.user.user_id],
        acl_inherited_grants=[],
    )
    await fs._get_vector_store()._upsert_many_raw([row], ctx=ctx)
    reader = RequestContext(user=UserIdentifier(ctx.account_id, "reader"), role=Role(Role.USER))
    await fs._ensure_access(TARGET, reader, action=document_write.AclAction.READ)
    if input_path == "physical":

        async def no_context(*args, **kwargs):
            return []

        monkeypatch.setattr(document_write, "_records", no_context)
    with pytest.raises(PermissionDeniedError) as raised:
        await document_node(service, TARGET, reader)
    error = raised.value
    assert "classified-child" not in str(error) + repr(error.details)
    assert error.__cause__ is None and error.__context__ is None
    app.dependency_overrides[get_request_context] = lambda: reader
    response = await client.get("/api/v1/resources/document-node", params={"uri": TARGET})
    assert response.status_code == 403, response.text
    assert "classified-child" not in response.text
    assert TARGET in response.text


async def test_duplicate_slot_controls_refuse_before_effects(admitted_service, monkeypatch):
    from openviking.service import document_write
    from openviking_cli.exceptions import FailedPreconditionError

    service, ctx = admitted_service, _root_ctx()
    fs = service.viking_fs
    await fs.write_file_bytes(TARGET, b"original", ctx=ctx)
    rows = []
    for index, tag in enumerate(("first", "second")):
        row = document_write._root_record(TARGET, 2, ctx, False)
        row.update(
            id="duplicate-control-" + str(index), vector=[0.1] * 2048, search_tags=[tag], tags=tag
        )
        rows.append(row)
    await fs._get_vector_store()._upsert_many_raw(rows, ctx=ctx)
    actual = await document_write._records(fs, TARGET, ctx)
    assert len(actual) == 2
    before = (await document_node(service, TARGET, ctx))["node"]
    effects = []

    async def forbidden(*args, **kwargs):
        effects.append(True)
        raise AssertionError("ambiguous controls must refuse before preparation/effects")

    monkeypatch.setattr(document_write, "_prepare", forbidden)
    monkeypatch.setattr(document_write, "_publish_records", forbidden)
    monkeypatch.setattr(fs, "write_file_bytes", forbidden)
    with pytest.raises(FailedPreconditionError) as raised:
        await write_document(
            service,
            DocumentWriteRequest(
                target_uri=TARGET,
                source_name="replacement.md",
                text="new",
                representation="raw_file",
                expected=before,
                backup_uri=BACKUP,
            ),
            ctx,
        )
    assert raised.value.details["content_state"] == "not_started"
    assert effects == []
    assert (await document_node(service, TARGET, ctx))["node"] == before
    assert await document_write._records(fs, TARGET, ctx) == actual
    assert await fs.read_file_bytes(TARGET, ctx=ctx) == b"original"
    assert (await document_node(service, BACKUP, ctx))["node"] == {"kind": "absent"}


async def test_backup_sidecars_map_and_rollback_exact(admitted_service, monkeypatch):
    from openviking.service import document_write
    from openviking.storage.abstract_overview import (
        parse_abstract_overview,
        render_abstract_overview,
        rewrite_abstract_overview_for_transfer,
    )

    service, ctx = admitted_service, _root_ctx()
    fs = service.viking_fs
    await fs.mkdir(TARGET, ctx=ctx)
    await fs.write_file_bytes(TARGET + "/original.md", b"original body", ctx=ctx)
    row = document_write._root_record(TARGET, 0, ctx, True)
    row.update(vector=[0.1] * 2048, search_tags=["retained"])
    await fs._get_vector_store().upsert(row, ctx=ctx)
    await service.runtime_config_manager.patch_account(ctx.account_id, {"acl": {"enabled": True}})
    await fs.set_acl(
        TARGET,
        [{"principal": "user:" + ctx.user.user_id, "level": "manage"}],
        acl_mode=AclMode.RESTRICTED,
        ctx=ctx,
    )
    relations = b'{"legacy": ["literal relation bytes"]}\n'
    await fs.write_file_bytes(TARGET + "/.relations.json", relations, ctx=ctx)
    originals = {}
    for level, name in ((0, ".abstract.md"), (1, ".overview.md")):
        raw = render_abstract_overview(level, TARGET, "[old](" + TARGET + "/original.md)\n", {})
        originals[name] = raw.encode()
        await fs.write_file_bytes(TARGET + "/" + name, raw.encode(), ctx=ctx)
    before = (await document_node(service, TARGET, ctx))["node"]
    old_records = await document_write._records(fs, TARGET, ctx)
    outsider = RequestContext(user=UserIdentifier(ctx.account_id, "outsider"), role=Role(Role.USER))
    write = fs.write_file_bytes
    checked = []

    async def protected_write(uri, data, *args, **kwargs):
        result = await write(uri, data, *args, **kwargs)
        if uri.startswith(TARGET + "/") or uri.startswith(BACKUP + "/"):
            with pytest.raises(PermissionDeniedError):
                await fs.read_file_bytes(uri, ctx=outsider)
            checked.append(uri)
        return result

    async def fault(phase):
        if phase == "install":
            raise OSError("one-shot install fault")

    monkeypatch.setattr(fs, "write_file_bytes", protected_write)
    with pytest.raises(OpenVikingError) as raised:
        await write_document(
            service,
            DocumentWriteRequest(
                target_uri=TARGET,
                source_name="replacement.md",
                text="# New\n\nNew body.",
                expected=before,
                backup_uri=BACKUP,
            ),
            ctx,
            fault=fault,
        )
    assert raised.value.details["content_state"] == "rolled_back", raised.value.details
    assert (await document_node(service, TARGET, ctx))["node"] == before
    assert await document_write._records(fs, TARGET, ctx) == old_records
    backup = (await document_node(service, BACKUP, ctx))["node"]
    assert backup == raised.value.details["retained_copies"][0]["node"]
    assert backup != before
    assert await fs.read_file_bytes(TARGET + "/.relations.json", ctx=ctx) == relations
    assert await fs.read_file_bytes(BACKUP + "/.relations.json", ctx=ctx) == relations
    assert TARGET + "/.relations.json" in checked and BACKUP + "/.relations.json" in checked
    for level, name in ((0, ".abstract.md"), (1, ".overview.md")):
        assert await fs.read_file_bytes(TARGET + "/" + name, ctx=ctx) == originals[name]
        actual = await fs.read_file_bytes(BACKUP + "/" + name, ctx=ctx)
        assert (
            actual
            == rewrite_abstract_overview_for_transfer(
                originals[name],
                level=level,
                source_dir_uri=TARGET,
                target_dir_uri=BACKUP,
                source_scope_uri=TARGET,
                target_scope_uri=BACKUP,
            ).encode()
        )
        parsed = parse_abstract_overview(actual)
        assert parsed.metadata["directory"] == BACKUP + "/"
        assert BACKUP + "/original.md" in parsed.body
        assert TARGET + "/original.md" not in parsed.body
        assert TARGET + "/" + name in checked and BACKUP + "/" + name in checked


@pytest.mark.parametrize(
    "suffix",
    ["/.relations.json", "/nested/.relations.json", "/.abstract.md", "/.overview.md"],
)
def test_snapshot_controls_are_not_direct_targets(service, suffix):
    from openviking.service.document_write import _uri
    from openviking.storage.resource_rfv import validate_resource_reindex_uri
    from openviking_cli.exceptions import InvalidArgumentError

    fs, ctx = service.viking_fs, _root_ctx()
    uri = TARGET + suffix
    assert _uri(fs, uri, ctx, generated=True) == uri
    with pytest.raises(InvalidArgumentError):
        _uri(fs, uri, ctx)
    if suffix.endswith("/.relations.json"):
        with pytest.raises(ValueError):
            validate_resource_reindex_uri(fs, uri, ctx)


@pytest.mark.parametrize(
    "suffix", ["/.relations.json/child.md", "/../.relations.json", "/%2e/.relations.json"]
)
def test_snapshot_sidecar_does_not_relax_path_grammar(service, suffix):
    from openviking.service.document_write import _uri
    from openviking_cli.exceptions import InvalidArgumentError

    with pytest.raises(InvalidArgumentError):
        _uri(service.viking_fs, TARGET + suffix, _root_ctx(), generated=True)


def test_snapshot_sidecar_requires_exact_storage_roundtrip(service, monkeypatch):
    from openviking.service.document_write import _uri
    from openviking_cli.exceptions import InvalidArgumentError

    fs = service.viking_fs
    monkeypatch.setattr(fs, "_path_to_uri", lambda *args, **kwargs: TARGET + "/other")
    with pytest.raises(InvalidArgumentError):
        _uri(fs, TARGET + "/.relations.json", _root_ctx(), generated=True)


def test_closed_wire_shapes():
    from pydantic import ValidationError

    base = {
        "target_uri": TARGET,
        "source_name": "note.md",
        "text": "",
        "expected": {"kind": "absent"},
    }
    assert DocumentWriteRequest(**base).text == ""
    for expected in (
        {"kind": "absent", "revision": "absent"},
        {"kind": "directory", "tree_revision": "0" * 64},
        {"kind": "file", "content_sha256": "0" * 64, "control_revision": "file:" + "0" * 64},
    ):
        with pytest.raises(ValidationError):
            DocumentWriteRequest(**dict(base, expected=expected))
