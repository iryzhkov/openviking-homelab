"""Native document-write vertical slices. Only model providers are synthetic."""

import asyncio

import pytest
from openviking.config.embedding import AccountBoundEmbedder
from openviking.server.auth import get_request_context
from openviking.server.identity import RequestContext, Role
from openviking.storage.acl import AclMode
from openviking.utils.media_processor import UnifiedResourceProcessor
from openviking_cli.exceptions import ConflictError, OpenVikingError, PermissionDeniedError
from openviking_cli.session.user_id import UserIdentifier

from openviking.service.document_write import DocumentWriteRequest, document_node, write_document
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
    await asyncio.wait_for(service.fs.drain_acked_transactions(), timeout=35)
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
    exact_contract = await client.get(
        "/api/v1/resources/document-write-contract", params={"version": 2}
    )
    assert exact_contract.json()["result"] == {"contract": "resources-document-write", "version": 2}
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
    assert result["processing_state"] in {"accepted", "unobserved"}, result
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
    assert replaced["processing_state"] in {"accepted", "unobserved"}
    assert replaced["cleanup_state"] in {"complete", "unknown"}
    await settle(service)
    assert excluded_links == []


async def test_exact_directory_original_uri_and_protected_backup(admitted_service):
    from pydantic import ValidationError

    service, ctx = admitted_service, _root_ctx()
    fs = service.viking_fs
    await fs.mkdir(TARGET, ctx=ctx)
    old = b"old exact\r\n "
    await fs.write_file_bytes(TARGET + "/old.md", old, ctx=ctx)
    sidecar = b'{"relations":[]}'
    await fs.write_file_bytes(TARGET + "/.relations.json", sidecar, ctx=ctx)
    from openviking.service import document_write

    row = document_write._root_record(TARGET, 0, ctx, True)
    row["vector"] = [0.1] * 2048
    await fs._get_vector_store().upsert(row, ctx=ctx)
    await service.runtime_config_manager.patch_account(ctx.account_id, {"acl": {"enabled": True}})
    await fs.set_acl(
        TARGET,
        [{"principal": "user:" + ctx.user.user_id, "level": "manage"}],
        acl_mode=AclMode.RESTRICTED,
        ctx=ctx,
    )
    expected = (await document_node(service, TARGET, ctx))["node"]
    body = "# Original Ω\n\n## Decisions 漢字\n\n" + ("  cobalt Ω  \t\n\n" * 650) + "\r\n "
    raw = body.encode("utf-8")
    assert len(raw) > len(body) > 6000
    values = {
        "target_uri": TARGET,
        "source_name": "note.md",
        "text": body,
        "representation": "exact_directory",
        "expected": expected,
        "backup_uri": BACKUP,
    }
    with pytest.raises(ValidationError):
        DocumentWriteRequest(**values)
    request = DocumentWriteRequest(**values, contract_version=2)
    outsider = RequestContext(user=UserIdentifier(ctx.account_id, "outsider"), role=Role(Role.USER))
    with pytest.raises(PermissionDeniedError):
        await write_document(service, request, outsider)
    assert (await document_node(service, BACKUP, ctx))["node"]["kind"] == "absent"
    # Sidecars participate in the directory CAS.
    await fs.write_file_bytes(TARGET + "/.relations.json", b'{"relations":["changed"]}', ctx=ctx)
    with pytest.raises(ConflictError):
        await write_document(service, request, ctx)
    assert (await document_node(service, BACKUP, ctx))["node"]["kind"] == "absent"
    current_sidecar = await fs.read_file_bytes(TARGET + "/.relations.json", ctx=ctx)
    values["expected"] = (await document_node(service, TARGET, ctx))["node"]
    result = await asyncio.wait_for(
        write_document(service, DocumentWriteRequest(**values, contract_version=2), ctx), timeout=60
    )
    assert result["uri"] == TARGET and result["representation"] == "exact_directory"
    assert result["content_state"] == "committed", result
    assert await fs.read_file_bytes(TARGET + "/0001/0000.md", ctx=ctx) == raw
    assert await fs.read_file_bytes(BACKUP + "/old.md", ctx=ctx) == old
    assert await fs.read_file_bytes(BACKUP + "/.relations.json", ctx=ctx) == current_sidecar
    for uri in (TARGET + "/0001/0000.md", BACKUP + "/old.md"):
        with pytest.raises(PermissionDeniedError):
            await fs.read_file_bytes(uri, ctx=outsider)
    await settle(service)


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
    from openviking.storage.vector_ids import vector_record_id
    from openviking_cli.exceptions import FailedPreconditionError

    from openviking.service import document_write

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
    from openviking_cli.exceptions import FailedPreconditionError

    from openviking.service import document_write

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
    from openviking.storage.abstract_overview import (
        parse_abstract_overview,
        render_abstract_overview,
        rewrite_abstract_overview_for_transfer,
    )

    from openviking.service import document_write

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
    from openviking.storage.resource_rfv import validate_resource_reindex_uri
    from openviking_cli.exceptions import InvalidArgumentError

    from openviking.service.document_write import _uri

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
    from openviking_cli.exceptions import InvalidArgumentError

    from openviking.service.document_write import _uri

    with pytest.raises(InvalidArgumentError):
        _uri(service.viking_fs, TARGET + suffix, _root_ctx(), generated=True)


def test_snapshot_sidecar_requires_exact_storage_roundtrip(service, monkeypatch):
    from openviking_cli.exceptions import InvalidArgumentError

    from openviking.service.document_write import _uri

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


@pytest.mark.parametrize("phase", ["capture", "enqueue"])
async def test_verified_ack_retains_transaction(admitted_service, monkeypatch, phase):
    from pathlib import Path

    from openviking.service import document_write as module

    service = admitted_service
    ctx = _root_ctx()
    expected = await seed_file(service, restricted=True)
    entered, release = asyncio.Event(), asyncio.Event()
    paths, leases, released = [], [], []
    original_temp = module.tempfile.TemporaryDirectory
    original_release = service.viking_fs._async_agfs.pathlock_release

    def temporary(*args, **kwargs):
        value = original_temp(*args, **kwargs)
        paths.append(Path(value.name))
        return value

    async def release_lease(lease, *args, **kwargs):
        released.append(lease)
        return await original_release(lease, *args, **kwargs)

    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", temporary)
    monkeypatch.setattr(service.viking_fs._async_agfs, "pathlock_release", release_lease)
    owner = module if phase == "capture" else service.resources
    name = (
        "capture_resource_index_conditions" if phase == "capture" else "_enqueue_add_resource_job"
    )
    original = getattr(owner, name)

    async def gated(*args, **kwargs):
        leases.append(args[-1] if phase == "capture" else kwargs["resource_lock"])
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(owner, name, gated)
    request = DocumentWriteRequest(
        target_uri=TARGET,
        source_name="note.txt",
        text="exact new\r\n",
        reason=HINT,
        expected=expected,
        representation="raw_file",
        backup_uri=BACKUP,
    )
    try:
        ack = await asyncio.wait_for(write_document(service, request, ctx), 20)
        await asyncio.wait_for(entered.wait(), 5)
        assert ack["content_state"] == "committed"
        assert ack["processing_state"] == "unobserved"
        assert ack["cleanup_state"] == "unknown"
        assert ack["backup"]["node"] == expected
        assert await service.viking_fs.read_file_bytes(TARGET, ctx=ctx) == b"exact new\r\n"
        assert await service.viking_fs.read_file_bytes(BACKUP, ctx=ctx) == b"original\n"
        assert len(service.fs._acked_transactions) == 1
        assert leases[0] not in released
        assert paths and all(path.exists() for path in paths)
    finally:
        release.set()
        await asyncio.wait_for(service.fs.drain_acked_transactions(), 20)
    assert len(leases) == 1
    assert not service.fs._acked_transactions
    assert all(not path.exists() for path in paths)
    assert ack["processing_state"] == "unobserved" and ack["cleanup_state"] == "unknown"
    await settle(service)


@pytest.mark.parametrize("phase", ["install", "verified"])
async def test_verified_ack_precommit_failure(admitted_service, phase):
    service = admitted_service
    ctx = _root_ctx()
    expected = await seed_file(service)

    async def fault(current):
        if current == phase:
            raise RuntimeError("synthetic precommit fault")

    with pytest.raises(OpenVikingError) as raised:
        await write_document(
            service,
            DocumentWriteRequest(
                target_uri=TARGET,
                source_name="note.txt",
                text="replacement",
                reason=HINT,
                expected=expected,
                representation="raw_file",
                backup_uri=BACKUP,
            ),
            ctx,
            fault=fault,
        )
    assert raised.value.details["content_state"] == "rolled_back"
    assert raised.value.details["processing_state"] == "unobserved"
    assert not service.fs._acked_transactions
    assert await service.viking_fs.read_file_bytes(TARGET, ctx=ctx) == b"original\n"


async def test_cancel_before_verified_finishes_without_ack(admitted_service):
    service = admitted_service
    ctx = _root_ctx()
    entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def fault(phase):
        if phase == "prepared":
            entered.set()
            await release.wait()

    async def caller():
        try:
            return await write_document(
                service,
                DocumentWriteRequest(
                    target_uri=TARGET,
                    source_name="note.md",
                    text="cancellation body",
                    reason=HINT,
                    expected={"kind": "absent"},
                ),
                ctx,
                fault=fault,
            )
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(caller())
    try:
        await asyncio.wait_for(entered.wait(), 20)
        task.cancel()
        # A scheduled barrier lets the caller handle cancellation without a timing sleep.
        barrier = asyncio.Event()
        asyncio.get_running_loop().call_soon(barrier.set)
        await barrier.wait()
        assert not task.done() and not cancelled.is_set()
        assert service.fs._acked_transactions
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 20)
    assert cancelled.is_set() and not service.fs._acked_transactions
    await settle(service)
    assert (await document_node(service, TARGET, ctx))["node"]["kind"] == "directory"


async def test_shutdown_drains_before_dependencies(admitted_service, monkeypatch):
    service = admitted_service
    ctx = _root_ctx()
    entered, release, closing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = service.resources._enqueue_add_resource_job

    async def gated(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    class StopClose(Exception):
        pass

    async def close_background():
        assert not service.fs._acked_transactions
        closing.set()
        raise StopClose()

    with monkeypatch.context() as patch:
        patch.setattr(service.resources, "_enqueue_add_resource_job", gated)
        ack = await asyncio.wait_for(
            write_document(
                service,
                DocumentWriteRequest(
                    target_uri=TARGET,
                    source_name="note.md",
                    text="shutdown body",
                    reason=HINT,
                    expected={"kind": "absent"},
                ),
                ctx,
            ),
            20,
        )
        assert ack["cleanup_state"] == "unknown"
        await asyncio.wait_for(entered.wait(), 5)
        patch.setattr(service.resources, "close_background_tasks", close_background)
        close = asyncio.create_task(service.close())
        try:
            barrier = asyncio.Event()
            asyncio.get_running_loop().call_soon(barrier.set)
            await barrier.wait()
            assert not close.done() and not closing.is_set()
            assert all(not task.cancelled() for task in service.fs._acked_transactions)
        finally:
            release.set()
        with pytest.raises(StopClose):
            await asyncio.wait_for(close, 20)
        assert closing.is_set()
    await settle(service)


async def test_real_asgi_ack_precedes_enqueue(admitted_service, app, client, monkeypatch):
    service = admitted_service
    ctx = _root_ctx()
    app.dependency_overrides[get_request_context] = lambda: ctx
    entered, release = asyncio.Event(), asyncio.Event()
    original = service.resources._enqueue_add_resource_job

    async def gated(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(service.resources, "_enqueue_add_resource_job", gated)
    try:
        response = await asyncio.wait_for(
            client.post(
                "/api/v1/resources/document-write",
                json={
                    "target_uri": TARGET,
                    "source_name": "note.md",
                    "text": "ASGI body",
                    "reason": HINT,
                    "expected": {"kind": "absent"},
                },
            ),
            20,
        )
        await asyncio.wait_for(entered.wait(), 5)
        assert response.status_code == 200, response.text
        ack = response.json()["result"]
        assert ack["content_state"] == "committed"
        assert ack["processing_state"] == "unobserved" and ack["cleanup_state"] == "unknown"
        assert service.fs._acked_transactions and not release.is_set()
    finally:
        release.set()
        await settle(service)


async def test_second_write_excluded_until_continuation(admitted_service, monkeypatch):
    from openviking.storage.errors import LockAcquisitionError

    service = admitted_service
    ctx = _root_ctx()
    expected = await seed_file(service)
    entered, release = asyncio.Event(), asyncio.Event()
    original = service.resources._enqueue_add_resource_job
    calls = []

    async def gated(*args, **kwargs):
        calls.append(kwargs["resource_lock"])
        if len(calls) == 1:
            entered.set()
            await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(service.resources, "_enqueue_add_resource_job", gated)
    first = await asyncio.wait_for(
        write_document(
            service,
            DocumentWriteRequest(
                target_uri=TARGET,
                source_name="note.txt",
                text="first body",
                reason=HINT,
                expected=expected,
                representation="raw_file",
            ),
            ctx,
        ),
        20,
    )
    await asyncio.wait_for(entered.wait(), 5)
    request = DocumentWriteRequest(
        target_uri=TARGET,
        source_name="note.txt",
        text="second body",
        reason=HINT,
        expected=first["node"],
        backup_uri=BACKUP,
        representation="raw_file",
    )
    try:
        # Native account coverage fails fast rather than queueing another lease.
        with pytest.raises(LockAcquisitionError):
            await asyncio.wait_for(write_document(service, request, ctx), 5)
        assert len(calls) == 1
        assert await service.viking_fs.read_file_bytes(TARGET, ctx=ctx) == b"first body"
        assert not await service.viking_fs.exists(BACKUP, ctx=ctx)
    finally:
        release.set()
        await settle(service)
    ack = await asyncio.wait_for(write_document(service, request, ctx), 20)
    assert ack["content_state"] == "committed"
    await settle(service)


async def test_after_ack_context_failure_observed_once(admitted_service, monkeypatch):
    from contextlib import asynccontextmanager
    from unittest.mock import Mock

    from openviking.service import fs_service as fs_module

    service = admitted_service
    ctx = _root_ctx()
    entered, release = asyncio.Event(), asyncio.Event()
    original_enqueue = service.resources._enqueue_add_resource_job
    original_mutation = service.fs._uri_mutation_coordinator.mutation
    log = Mock()
    loop_errors = []
    loop = asyncio.get_running_loop()
    original_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: loop_errors.append(context))

    async def gated(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original_enqueue(*args, **kwargs)

    @asynccontextmanager
    async def mutation(*args, **kwargs):
        async with original_mutation(*args, **kwargs):
            yield
        raise RuntimeError("SECRET body/URI must not reach logs")

    monkeypatch.setattr(service.resources, "_enqueue_add_resource_job", gated)
    monkeypatch.setattr(service.fs._uri_mutation_coordinator, "mutation", mutation)
    monkeypatch.setattr(fs_module.logger, "error", log)
    try:
        ack = await asyncio.wait_for(
            write_document(
                service,
                DocumentWriteRequest(
                    target_uri=TARGET,
                    source_name="note.md",
                    text="failure body",
                    reason=HINT,
                    expected={"kind": "absent"},
                ),
                ctx,
            ),
            20,
        )
        await asyncio.wait_for(entered.wait(), 5)
        assert ack["content_state"] == "committed"
    finally:
        release.set()
        await asyncio.wait_for(service.fs.drain_acked_transactions(), 20)
        loop.set_exception_handler(original_handler)
    assert log.call_count == 1
    assert log.call_args.args == (
        "Filesystem %s continuation failed: %s",
        "document-write",
        "OpenVikingError",
    )
    assert not loop_errors and not service.fs._acked_transactions
    await settle(service)


@pytest.mark.parametrize("phase", ["capture", "enqueue"])
async def test_after_ack_caught_postcommit_failure(admitted_service, monkeypatch, phase):
    from unittest.mock import Mock

    from openviking.service import document_write as module

    service = admitted_service
    ctx = _root_ctx()
    entered, release = asyncio.Event(), asyncio.Event()
    owner = module if phase == "capture" else service.resources
    name = (
        "capture_resource_index_conditions" if phase == "capture" else "_enqueue_add_resource_job"
    )
    original = getattr(owner, name)
    log = Mock()
    logger = Mock(error=log)
    monkeypatch.setattr(module, "get_logger", lambda name: logger)

    async def fail(*args, **kwargs):
        entered.set()
        await release.wait()
        # Exercise the real queue path before a synthetic response failure.
        await original(*args, **kwargs)
        raise RuntimeError("SECRET document body")

    monkeypatch.setattr(owner, name, fail)
    try:
        ack = await asyncio.wait_for(
            write_document(
                service,
                DocumentWriteRequest(
                    target_uri=TARGET,
                    source_name="note.md",
                    text="postcommit failure",
                    reason=HINT,
                    expected={"kind": "absent"},
                ),
                ctx,
            ),
            20,
        )
        await asyncio.wait_for(entered.wait(), 5)
        assert ack["processing_state"] == "unobserved"
    finally:
        release.set()
        await asyncio.wait_for(service.fs.drain_acked_transactions(), 20)
    assert not service.fs._acked_transactions
    log.assert_called_once_with("Document postcommit continuation failed: %s", "RuntimeError")
    assert ack["cleanup_state"] == "unknown"
    await settle(service)


async def test_verified_ack_prefers_finished_result(admitted_service):
    service = admitted_service
    verified = asyncio.get_running_loop().create_future()

    async def transaction():
        verified.set_result({"cleanup_state": "unknown"})
        return {"cleanup_state": "complete", "processing_state": "accepted"}

    result = await service.fs._finish_transfer_after_verified_ack(
        transaction,
        verified,
        operation="document-write",
    )
    assert result == {"cleanup_state": "complete", "processing_state": "accepted"}
    assert not service.fs._acked_transactions


@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_document_capacity_held_until_transaction_done(
    admitted_service, monkeypatch, recwarn, cancel_waiter
):
    import gc

    from openviking.service import document_write as module

    service = admitted_service
    ctx = _root_ctx()
    capacity = asyncio.BoundedSemaphore(1)
    monkeypatch.setattr(service.fs, "_document_transactions", capacity)
    entered, release, attempted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_enqueue = service.resources._enqueue_add_resource_job
    original_write = module._write_document
    started = []

    async def observed(*args, **kwargs):
        started.append(args[1].target_uri)
        return await original_write(*args, **kwargs)

    async def gated(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            await release.wait()
        result = await original_enqueue(*args, **kwargs)
        # Let the real queue owner release its account coverage before admission.
        await asyncio.wait_for(service.resources.wait_processed(), 20)
        return result

    monkeypatch.setattr(module, "_write_document", observed)
    monkeypatch.setattr(service.resources, "_enqueue_add_resource_job", gated)
    second_uri = BASE + "/second.md"

    async def second_call():
        attempted.set()
        return await write_document(
            service,
            DocumentWriteRequest(
                target_uri=second_uri,
                source_name="second.md",
                text="second body",
                reason=HINT,
                expected={"kind": "absent"},
            ),
            ctx,
        )

    second = None
    try:
        ack = await asyncio.wait_for(
            write_document(
                service,
                DocumentWriteRequest(
                    target_uri=TARGET,
                    source_name="note.md",
                    text="first body",
                    reason=HINT,
                    expected={"kind": "absent"},
                ),
                ctx,
            ),
            20,
        )
        await asyncio.wait_for(entered.wait(), 5)
        assert ack["content_state"] == "committed"
        assert ack["cleanup_state"] == "unknown"
        second = asyncio.create_task(second_call())
        await asyncio.wait_for(attempted.wait(), 5)
        assert not second.done()
        assert started == [TARGET]
        assert len(service.fs._acked_transactions) == 1
        assert not await service.viking_fs.exists(second_uri, ctx=ctx)
        if cancel_waiter:
            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(second, 5)
            assert started == [TARGET]
            assert len(service.fs._acked_transactions) == 1
            gc.collect()
            assert not [w for w in recwarn if "was never awaited" in str(w.message)]
        release.set()
        if not cancel_waiter:
            result = await asyncio.wait_for(second, 20)
            assert result["content_state"] == "committed"
            assert started == [TARGET, second_uri]
    finally:
        release.set()
        if second is not None and not second.done():
            await asyncio.wait_for(second, 20)
        await settle(service)
    assert not service.fs._acked_transactions
    # The one permit returns exactly once, after the callback, even after ACK.
    await asyncio.wait_for(capacity.acquire(), 5)
    assert capacity.locked()
    capacity.release()
    if cancel_waiter:
        assert not await service.viking_fs.exists(second_uri, ctx=ctx)
        gc.collect()
        assert not [w for w in recwarn if "was never awaited" in str(w.message)]
