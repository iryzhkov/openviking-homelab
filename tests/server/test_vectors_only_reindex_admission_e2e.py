"""Fresh vectors-only reindex through persisted jobs and real fenced handlers."""

import asyncio
import json

import pytest

from openviking.storage.queuefs.named_queue import NamedQueue
from tests.server.conftest import service as service
from tests.server.conftest import temp_dir as temp_dir
from tests.server.test_real_resource_producer_e2e import (
    _creation_identity,
    _records,
    _recreate_account,
    _root_ctx,
)
from tests.server.test_real_resource_producer_e2e import (
    admitted_service as admitted_service,
)


@pytest.fixture
def persisted(monkeypatch):
    entries = []
    original = NamedQueue.enqueue

    async def record(self, data):
        entries.append((self.name, json.loads(json.dumps(data))))
        return await original(self, data)

    monkeypatch.setattr(NamedQueue, "enqueue", record)
    return entries


async def _seed(service, *, indexed=False):
    ctx = _root_ctx()
    root = "viking://resources/vectors-reindex"
    uri = root + "/note.md"
    if indexed:
        await service.fs.write(
            uri,
            "# Original source\n\nPreserved content",
            ctx=ctx,
            wait=True,
            tags=["team=admission"],
        )
    else:
        await service.viking_fs.mkdir(root, ctx=ctx)
        await service.viking_fs.write_file(uri, "# Original source\n\nPreserved content", ctx=ctx)
    return root, uri


async def _run(service, uri):
    return await asyncio.wait_for(
        service.reindex(uri=uri, mode="vectors_only", force=True, wait=True, ctx=_root_ctx()),
        timeout=25,
    )


def _forbid_models(monkeypatch):
    from openviking_cli.utils.config.vlm_config import VLMConfig

    async def forbidden(*args, **kwargs):
        raise AssertionError("vectors_only called a generation model")

    monkeypatch.setattr(VLMConfig, "get_completion_async", forbidden)


@pytest.mark.parametrize("directory", [False, True])
async def test_fresh_vectors_only_reindex_persisted_original_conditions(
    admitted_service, monkeypatch, persisted, directory
):
    service = admitted_service
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    root, uri = await _seed(service)
    original_identity = await _creation_identity(service)
    original_bytes = await service.viking_fs.read_file_bytes(uri, ctx=_root_ctx())
    _forbid_models(monkeypatch)
    result = await _run(service, root if directory else uri)
    assert result["failed_records"] == 0, result
    assert result["rebuilt_records"] >= 1, result
    assert await service.viking_fs.read_file_bytes(uri, ctx=_root_ctx()) == original_bytes
    jobs = [d for name, d in persisted if "eindex" in name]
    embeddings = [d for name, d in persisted if "mbedding" in name]
    assert jobs and embeddings, persisted
    assert all(d["resource_creation_identity"] == original_identity for d in jobs)
    assert not [d for name, d in persisted if "emantic" in name]
    for message in embeddings:
        proof = message["payload"]["context_data"]["_resource_fence"]
        assert proof["creation_identity"] == original_identity
        assert proof["owner"]
    records = await _records(service, _root_ctx())
    assert (uri, 2) in {(r.get("uri"), r.get("level")) for r in records}, records


async def test_vectors_only_preserves_sidecars_and_tags(admitted_service, monkeypatch, persisted):
    service = admitted_service
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    root, uri = await _seed(service, indexed=True)
    before = {
        name: await service.viking_fs.read_file_bytes(root + "/" + name, ctx=_root_ctx())
        for name in ["note.md", ".abstract.md", ".overview.md"]
    }
    persisted.clear()
    _forbid_models(monkeypatch)
    result = await _run(service, root)
    assert result["failed_records"] == 0, result
    assert result["rebuilt_records"] >= 3, result
    for name, raw in before.items():
        assert await service.viking_fs.read_file_bytes(root + "/" + name, ctx=_root_ctx()) == raw
    records = await _records(service, _root_ctx())
    detail = next(r for r in records if r.get("uri") == uri and r.get("level") == 2)
    assert "team=admission" in detail["search_tags"], detail
    assert not [d for name, d in persisted if "emantic" in name]


async def test_vectors_only_orphan_delete_uses_fenced_queue(
    admitted_service, monkeypatch, persisted
):
    service = admitted_service
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    root, uri = await _seed(service)
    await _run(service, root)
    # Create a genuine orphan without invoking VikingFS's index cleanup.
    await service.viking_fs._async_agfs.rm(service.viking_fs._uri_to_path(uri, ctx=_root_ctx()))
    persisted.clear()
    result = await _run(service, root)
    assert result["deleted_records"] >= 1, result
    deletes = [d for name, d in persisted if "mbedding" in name and d.get("action") == "delete"]
    assert deletes, persisted
    assert all(
        d["payload"]["context_data"]["_resource_fence"]["action"] == "delete" for d in deletes
    )
    assert not [r for r in await _records(service, _root_ctx()) if r.get("uri") == uri]


async def test_reindex_recreated_account_refuses_before_consumer_adoption(
    admitted_service, monkeypatch, persisted
):
    from openviking.storage.queuefs.reindex_processor import ReindexProcessor

    service = admitted_service
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    root, uri = await _seed(service)
    original = ReindexProcessor._on_dequeue_bound
    changed = []

    async def recreate(self, data):
        if not changed:
            await _recreate_account(service)
            changed.append(True)
        return await original(self, data)

    monkeypatch.setattr(ReindexProcessor, "_on_dequeue_bound", recreate)
    with pytest.raises(Exception, match="lifetime|identity|changed"):
        await _run(service, root)
    assert changed
    assert not [d for name, d in persisted if "mbedding" in name]
    assert not [r for r in await _records(service, _root_ctx()) if r.get("uri") == uri]


@pytest.mark.parametrize("change", ["source", "record"])
async def test_reindex_final_conditions_refuse_changed_inputs(
    admitted_service, monkeypatch, change
):
    from contextlib import asynccontextmanager

    from openviking.storage import collection_schemas

    service = admitted_service
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    root, uri = await _seed(service, indexed=True)
    backend = service.viking_fs._get_vector_store()
    original_upsert = backend.upsert
    writes = []

    async def record_write(*args, **kwargs):
        writes.append(args)
        return await original_upsert(*args, **kwargs)

    monkeypatch.setattr(backend, "upsert", record_write)
    original_commit = collection_schemas.resource_embedding_commit
    injected = []

    @asynccontextmanager
    async def mutate_then_commit(msg, ctx, db):
        if msg.context_data.get("uri") == uri and not injected:
            proof = msg.context_data["_resource_fence"]
            if change == "source":
                await service.viking_fs.write_file(
                    uri, "# Changed after capture", ctx=ctx, lease_ref=proof["owner"]
                )
            else:
                records = await _records(service, ctx)
                record = dict(
                    next(r for r in records if r.get("uri") == uri and r.get("level") == 2)
                )
                record["search_tags"] = ["injected=record-change"]
                record["vector"] = [0.1] * 2048
                await original_upsert(record, ctx=ctx)
            injected.append(True)
        async with original_commit(msg, ctx, db):
            yield

    monkeypatch.setattr(collection_schemas, "resource_embedding_commit", mutate_then_commit)
    result = await _run(service, uri)
    assert injected
    assert result["failed_records"] >= 1, result
    assert not writes, writes


async def test_lost_reindex_handoff_preserves_foreign_owner(
    admitted_service, monkeypatch, persisted
):
    from openviking.storage.queuefs.reindex_processor import ReindexProcessor

    service = admitted_service
    fs = service.viking_fs
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    root, uri = await _seed(service)
    original = ReindexProcessor._adopt_or_reacquire
    foreign = []

    async def replace_original_owner(self, msg, ctx):
        if not foreign:
            owned = await fs._async_agfs.pathlock_adopt(msg.lock_handoff)
            await fs._async_agfs.pathlock_release(owned)
            foreign.append(
                await fs._async_agfs.pathlock_acquire_tree(fs._uri_to_path(root, ctx=ctx))
            )
        return await original(self, msg, ctx)

    monkeypatch.setattr(ReindexProcessor, "_adopt_or_reacquire", replace_original_owner)
    try:
        with pytest.raises(Exception, match="handoff|lease|lock"):
            await _run(service, root)
        assert await fs._async_agfs.pathlock_refresh(foreign[0]) == "refreshed"
        assert not [d for name, d in persisted if "mbedding" in name]
    finally:
        if foreign:
            await fs._async_agfs.pathlock_release(foreign[0])


@pytest.mark.parametrize("tamper", [False, True])
async def test_reindex_tag_patch_retains_original_intent(admitted_service, monkeypatch, tamper):
    from dataclasses import replace

    from openviking.storage.index_action import FieldPatch, IndexAction
    from openviking.utils.resource_processor import ResourceProcessor

    service = admitted_service
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    root, uri = await _seed(service, indexed=True)
    _forbid_models(monkeypatch)
    changed = []
    original_enqueue = ResourceProcessor._enqueue_index_actions

    async def enqueue_with_original_or_changed_intent(self, actions, **kwargs):
        if tamper:
            altered = []
            for action in actions:
                if action.action is IndexAction.UPDATE_FIELDS:
                    action = replace(
                        action,
                        field_patch=FieldPatch(
                            {"search_tags": ["tampered=intent"]},
                            seed_fields=action.field_patch.seed_fields,
                        ),
                    )
                    changed.append(True)
                altered.append(action)
            actions = tuple(altered)
        return await original_enqueue(self, actions, **kwargs)

    monkeypatch.setattr(
        ResourceProcessor, "_enqueue_index_actions", enqueue_with_original_or_changed_intent
    )

    async def reindex():
        return await asyncio.wait_for(
            service.reindex(
                uri=uri,
                mode="vectors_only",
                force=False,
                tags=["team=approved"],
                tag_mode="append",
                wait=True,
                ctx=_root_ctx(),
            ),
            timeout=25,
        )

    if tamper:
        with pytest.raises(Exception, match="intent changed"):
            await reindex()
        assert changed
    else:
        result = await reindex()
        assert result["failed_records"] == 0, result
    records = await _records(service, _root_ctx())
    detail = next(r for r in records if r.get("uri") == uri and r.get("level") == 2)
    assert "tampered=intent" not in detail["search_tags"], detail
    assert ("team=approved" in detail["search_tags"]) is not tamper
