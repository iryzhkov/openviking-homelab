# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Real ResourceProcessor producer pipeline, end to end.

The request runs through ResourceService.add_resource and the real
ResourceProcessor (parse, tree commit, owned lease, prepared plan,
_process_prepared, _enqueue_index_actions, Summarizer), real persisted queues,
real semantic/embedding handlers and the local persistent vector store. Only
the model providers are synthetic (tests.server.conftest fakes).

Method wrappers below only record calls and always call the original.
"""

import json

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.queuefs.named_queue import NamedQueue
from openviking.utils.resource_processor import ResourceProcessor
from openviking_cli.session.user_id import UserIdentifier
from tests.server.conftest import sample_markdown_file as sample_markdown_file
from tests.server.conftest import service as service
from tests.server.conftest import temp_dir as temp_dir


@pytest.fixture
def trace(monkeypatch):
    calls = []
    persisted = []
    from openviking.utils.summarizer import Summarizer

    for owner, name in (
        (ResourceProcessor, "process_resource"),
        (ResourceProcessor, "finish_prepared_resource"),
        (ResourceProcessor, "_enqueue_index_actions"),
        (Summarizer, "summarize"),
    ):
        original = getattr(owner, name)

        async def record(self, *args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            return await _original(self, *args, **kwargs)

        monkeypatch.setattr(owner, name, record)
    original_enqueue = NamedQueue.enqueue

    async def record_enqueue(self, data):
        persisted.append((getattr(self, "name", ""), json.loads(json.dumps(data))))
        return await original_enqueue(self, data)

    monkeypatch.setattr(NamedQueue, "enqueue", record_enqueue)
    return calls, persisted


@pytest.fixture
async def admitted_service(service):
    """Attach the real FileStore account registry, as server startup does.

    Server startup calls ``viking_fs.set_account_identity_manager`` with its
    API key manager; the in-process dev-auth service fixture has none, and
    qualified resource operations refuse without it.
    """
    from openviking.server.api_keys.legacy import FileStore

    manager = FileStore(service.viking_fs, watch_enabled=False)
    await manager.load()
    service.viking_fs.set_account_identity_manager(manager)
    try:
        yield service
    finally:
        await manager.close()


def _root_ctx():
    return RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)


async def _records(service, ctx):
    store = service.viking_fs._get_vector_store()
    records, _ = await store.scroll(limit=200, ctx=ctx)
    return records


async def _ingest(service, sample_markdown_file, trace):
    calls, persisted = trace
    ctx = _root_ctx()
    result = await service.resources.add_resource(
        path=str(sample_markdown_file), ctx=ctx, reason="e2e", wait=True
    )
    root_uri = result.get("root_uri", "") or "viking://resources/sample"
    records = await _records(service, ctx)
    resource_records = [r for r in records if str(r.get("uri", "")).startswith(root_uri)]
    semantic = [data for name, data in persisted if "emantic" in name]
    embedding = [data for name, data in persisted if "mbedding" in name]
    return {
        "result": {k: v for k, v in result.items() if k != "_post_process"},
        "root_uri": root_uri,
        "semantic_uris": [(d.get("uri"), d.get("generation_trigger")) for d in semantic],
        "calls": calls,
        "queues": sorted({name for name, _ in persisted}),
        "semantic_identity": [d.get("resource_creation_identity") for d in semantic],
        "embedding_fenced": ["_resource_fence" in json.dumps(d) for d in embedding],
        "records": sorted((r.get("uri"), r.get("level")) for r in resource_records),
    }


async def _creation_identity(service):
    identity = await service.viking_fs._capture_resource_identity(_root_ctx())
    return identity.creation_identity


async def _recreate_account(service):
    from uuid import uuid4

    from openviking.server.api_keys.legacy import ACCOUNTS_PATH

    agfs = service.viking_fs._async_agfs
    data = json.loads(await agfs.cat(ACCOUNTS_PATH, fs_ctx={"bypass_cache": "true"}))
    data["accounts"][_root_ctx().account_id]["creation_id"] = str(uuid4())
    await agfs.write(ACCOUNTS_PATH, json.dumps(data).encode())


def _trace_locks(service, monkeypatch):
    """Record native lease calls (always calling through) for failure detail."""
    import asyncio

    agfs = service.viking_fs._async_agfs
    events = []

    def short(value):
        text = json.dumps(value, default=str)
        return text[:160]

    for name in (
        "pathlock_acquire_batch",
        "pathlock_acquire_tree",
        "pathlock_acquire_exact",
        "pathlock_adopt",
        "pathlock_handoff",
        "pathlock_release",
    ):
        original = getattr(agfs, name, None)
        if original is None:
            continue

        async def wrapped(*args, _original=original, _name=name, **kwargs):
            task = asyncio.current_task()
            label = task.get_name() if task else ""
            try:
                value = await _original(*args, **kwargs)
            except Exception as error:
                events.append((_name, label, short(args), short(kwargs), "ERR " + str(error)))
                raise
            events.append((_name, label, short(args), short(kwargs), short(value)))
            return value

        monkeypatch.setattr(agfs, name, wrapped)
    return events


async def test_real_resource_ingest_reaches_actual_handlers(
    admitted_service, sample_markdown_file, trace
):
    """Phases that do work today: producer chain, original identity, fenced vectors."""
    service = admitted_service
    summary = await _ingest(service, sample_markdown_file, trace)
    detail = json.dumps(summary, default=str)
    assert summary["calls"][:3] == [
        "process_resource",
        "finish_prepared_resource",
        "summarize",
    ], detail
    assert {"AddResource", "Semantic", "Embedding"} <= set(summary["queues"]), detail
    original = await _creation_identity(service)
    assert summary["semantic_identity"] and set(summary["semantic_identity"]) == {original}, detail
    assert summary["embedding_fenced"] and all(summary["embedding_fenced"]), detail
    # At least one fenced resource vector reached the local persistent store.
    assert summary["records"], detail


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PRODUCT: concurrent resource embedding fences each take resources-root "
        "tree coverage via _acquire_resource_coverage, whose "
        "pathlock_acquire_batch has no timeout (fails at 0 ms). The overlapping "
        "acquisition refuses, the vector write fails ('lock acquire timed out "
        "after 0ms, uri=viking://resources/sample'), and add_resource returns "
        "status=error with no root_uri."
    ),
)
async def test_real_resource_ingest_succeeds_with_root_uri(
    admitted_service, sample_markdown_file, trace, monkeypatch
):
    locks = _trace_locks(admitted_service, monkeypatch)
    summary = await _ingest(admitted_service, sample_markdown_file, trace)
    summary["locks"] = locks
    detail = json.dumps(summary, default=str)
    assert summary["result"].get("root_uri"), detail
    assert summary["result"].get("status") == "success", detail
    root = summary["root_uri"]
    assert (root, 0) in summary["records"] and (root, 1) in summary["records"], detail


async def test_account_recreated_before_semantic_processing_refuses_publication(
    admitted_service, sample_markdown_file, trace, monkeypatch
):
    from openviking.storage.queuefs.semantic_work import ResourceSemanticMessageWork

    service = admitted_service
    original = ResourceSemanticMessageWork.acquire_lock
    changed = []

    async def recreate_then_acquire(self, ctx):
        if not changed:
            await _recreate_account(service)
            changed.append(self.msg.uri)
        return await original(self, ctx)

    monkeypatch.setattr(ResourceSemanticMessageWork, "acquire_lock", recreate_then_acquire)
    summary = await _ingest(service, sample_markdown_file, trace)
    detail = json.dumps(summary, default=str)
    assert changed, detail
    assert summary["semantic_identity"] and all(summary["semantic_identity"]), detail
    assert summary["result"].get("status") != "success", detail
    assert (summary["root_uri"], 0) not in summary["records"], detail
    # Semantic work refuses before generating directory summaries.
    assert (summary["root_uri"], 1) not in summary["records"], detail
