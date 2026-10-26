"""Native document-write vertical slices. Only model providers are synthetic."""

import asyncio

import pytest

from openviking.server.auth import get_request_context
from openviking.server.identity import RequestContext, Role
from openviking.service.document_write import DocumentWriteRequest, document_node, write_document
from openviking.storage.acl import AclMode
from openviking.utils.media_processor import UnifiedResourceProcessor
from openviking_cli.exceptions import ConflictError, OpenVikingError, PermissionDeniedError
from openviking_cli.session.user_id import UserIdentifier
from tests.server.conftest import app as app
from tests.server.conftest import client as client
from tests.server.conftest import service as service
from tests.server.conftest import temp_dir as temp_dir
from tests.server.test_real_resource_producer_e2e import _root_ctx
from tests.server.test_real_resource_producer_e2e import admitted_service as admitted_service

BASE = "viking://resources"
TARGET = BASE + "/document.md"
BACKUP = BASE + "/saved.md"
HINT = "Process the literal cobalt decision."


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

    def fail_embedding(*args, **kwargs):
        raise RuntimeError("injected private embedding failure")

    with monkeypatch.context() as patch:
        patch.setattr(type(service.viking_fs.query_embedder), "embed", fail_embedding)
        with pytest.raises(OpenVikingError) as embedding_error:
            await write_document(service, request, ctx)
        assert embedding_error.value.details["content_state"] == "not_started"
        assert (await document_node(service, TARGET, ctx))["node"] == expected
        assert (await document_node(service, BACKUP, ctx))["node"]["kind"] == "absent"

    async def fault(phase):
        if phase == "install":
            raise OSError("injected install failure")

    with pytest.raises(OpenVikingError) as raised:
        await write_document(service, request, ctx, fault=fault)
    assert raised.value.details["content_state"] == "rolled_back", raised.value.details
    assert raised.value.details["retained_copies"][0]["uri"] == BACKUP
    assert (await document_node(service, TARGET, ctx))["node"] == expected


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
