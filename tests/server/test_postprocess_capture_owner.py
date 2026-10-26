# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Raw-text POST_PROCESS admission through real queues/native/store; synthetic models only."""

import asyncio
import json

import pytest

from openviking.service.task_tracker import get_task_tracker
from openviking.storage.queuefs.add_resource_processor import AddResourceProcessor
from openviking.storage.queuefs.named_queue import NamedQueue
from tests.server.conftest import service as service
from tests.server.conftest import temp_dir as temp_dir
from tests.server.test_real_resource_producer_e2e import _records, _root_ctx
from tests.server.test_real_resource_producer_e2e import admitted_service as admitted_service

RAW = "# Original capture\n\nMultiline raw text for deferred direct indexing.\n"
TARGET = "viking://resources/capture-owner"


async def test_raw_text_post_process_keeps_capture_owner(admitted_service, monkeypatch):
    service = admitted_service
    fs = service.viking_fs
    ctx = _root_ctx()
    seen = []
    adopted = []
    queued = []
    original_consume = AddResourceProcessor.on_dequeue
    original_adopt = fs._async_agfs.pathlock_adopt
    original_enqueue = NamedQueue.enqueue

    async def consume(processor, data):
        payload = data.get("data", data)
        if isinstance(payload, str):
            payload = json.loads(payload)
        if payload.get("job_phase") == "post_process":
            seen.append(json.loads(json.dumps(payload)))
        return await original_consume(processor, data)

    async def adopt(handoff):
        owner = await original_adopt(handoff)
        adopted.append((dict(handoff), dict(owner)))
        return owner

    async def enqueue(queue, data):
        if getattr(queue, "name", "") == "Embedding":
            queued.append(json.loads(json.dumps(data)))
        return await original_enqueue(queue, data)

    monkeypatch.setattr(AddResourceProcessor, "on_dequeue", consume)
    monkeypatch.setattr(fs._async_agfs, "pathlock_adopt", adopt)
    monkeypatch.setattr(NamedQueue, "enqueue", enqueue)
    result = await asyncio.wait_for(
        service.resources.add_resource(
            path=RAW,
            to=TARGET,
            ctx=ctx,
            build_index=True,
            processing_mode="vectors_only",
            wait=True,
        ),
        timeout=40,
    )
    assert seen, result
    payload = seen[0]
    prepared = payload["prepared"]
    actions = prepared["context_update_plan"]["direct_index_actions"]
    conditions = prepared["resource_index_conditions"]
    assert actions and conditions, prepared
    original_owner = next(iter(conditions.values()))["owner"]
    handoff, live_owner = next(pair for pair in adopted if pair[0] == payload["lock_handoff"])
    assert original_owner["lease_ref"] == handoff["lease_ref"]
    assert original_owner["lease_ref"] != live_owner["lease_ref"]
    assert original_owner["ownership_ref"] != live_owner["ownership_ref"]
    assert result.get("status") == "success", result
    task = await get_task_tracker().get(
        payload["task_id"], account_id=ctx.account_id, user_id=ctx.user.user_id
    )
    assert task is not None and task.status.value == "completed", task
    records = [
        r
        for r in await _records(service, ctx)
        if r.get("level") == 2 and str(r.get("uri", "")).startswith(TARGET)
    ]
    assert records, result
    for record in records:
        assert await fs.read_file_bytes(record["uri"], ctx=ctx) == RAW.encode()
    assert queued
    for data in queued:
        body = data.get("payload", data)
        if isinstance(body, str):
            body = json.loads(body)
        proof = body.get("context_data", {}).get("_resource_fence")
        assert proof and proof in conditions.values(), data
    try:
        status = await fs._async_agfs.pathlock_refresh(live_owner)
    except (RuntimeError, ValueError):
        status = "lost"
    assert status == "lost"


@pytest.mark.parametrize(
    "change",
    ["source", "record", "intent", "uuid", "lost", "foreign", "missing", "forged"],
)
async def test_real_post_process_refuses_changed_capture(admitted_service, monkeypatch, change):
    from contextlib import suppress

    from openviking.storage.queuefs.process_result import ProcessOutcome
    from openviking.utils.resource_processor import ResourceProcessor
    from tests.server.test_queued_admission_repair import _processor
    from tests.server.test_real_resource_producer_e2e import _recreate_account

    service = admitted_service
    fs = service.viking_fs
    agfs = fs._async_agfs
    ctx = _root_ctx()
    held = []
    original_enqueue = NamedQueue.enqueue

    async def hold(queue, data):
        if getattr(queue, "name", "") == "AddResource":
            held.append(json.loads(json.dumps(data)))
            return "held-for-capture-fault"
        return await original_enqueue(queue, data)

    monkeypatch.setattr(NamedQueue, "enqueue", hold)
    result = await service.resources.add_resource(
        path=RAW, to=TARGET, ctx=ctx, build_index=True, processing_mode="vectors_only", wait=False
    )
    monkeypatch.setattr(NamedQueue, "enqueue", original_enqueue)
    assert held, result
    payload = held[0].get("data", held[0])
    if isinstance(payload, str):
        payload = json.loads(payload)
    assert payload["job_phase"] == "post_process"
    prepared = payload["prepared"]
    actions = prepared["context_update_plan"]["direct_index_actions"]
    assert actions
    conditions = json.loads(json.dumps(prepared["resource_index_conditions"]))
    original_handoff = dict(payload["lock_handoff"])
    uri = actions[0]["uri"]
    record_id = actions[0]["record_id"]
    original_bytes = await fs.read_file_bytes(uri, ctx=ctx)
    assert original_bytes == RAW.encode()
    foreign_handoff = None
    if change == "intent":
        actions[0]["action"] = "delete"
    elif change == "uuid":
        await _recreate_account(service)
    elif change == "lost":
        owner = await agfs.pathlock_adopt(original_handoff)
        await agfs.pathlock_release(owner)
    elif change == "foreign":
        foreign = await agfs.pathlock_acquire_tree(
            fs._uri_to_path("viking://resources/foreign", ctx=ctx)
        )
        foreign_handoff = await agfs.pathlock_to_handoff(foreign)
        await agfs.pathlock_handoff(foreign)
        payload["lock_handoff"] = foreign_handoff
    elif change == "missing":
        for proof in prepared["resource_index_conditions"].values():
            proof.pop("owner")
    elif change == "forged":
        for proof in prepared["resource_index_conditions"].values():
            proof["owner"]["lease_ref"] = "forged-original-owner"

    original_finish = ResourceProcessor.finish_prepared_resource
    changed = []

    async def finish(processor, plan, *, resource_lock=None, **kwargs):
        if change == "source":
            await agfs.write(
                fs._uri_to_path(uri, ctx=ctx),
                b"changed source",
                fs_ctx={"account_id": ctx.account_id, "lease_ref": resource_lock["lease_ref"]},
            )
            changed.append(True)
        elif change == "record":
            await fs._get_vector_store().upsert(
                {
                    "id": record_id,
                    "uri": uri,
                    "account_id": ctx.account_id,
                    "level": 2,
                    "abstract": "intervening record",
                    "vector": [0.3] * 2048,
                    "search_tags": ["owner=retain"],
                },
                ctx=ctx,
            )
            changed.append(True)
        return await original_finish(processor, plan, resource_lock=resource_lock, **kwargs)

    monkeypatch.setattr(ResourceProcessor, "finish_prepared_resource", finish)
    payload["lock_handoff_retry"] = 2
    try:
        outcome = await asyncio.wait_for(
            _processor(service).on_dequeue({"data": json.dumps(payload)}), timeout=35
        )
        tracker = get_task_tracker()
        if change in {"source", "record"}:
            assert changed
            await asyncio.wait_for(
                tracker.wait(
                    payload["task_id"], account_id=ctx.account_id, user_id=ctx.user.user_id
                ),
                timeout=35,
            )
            task = await tracker.get(
                payload["task_id"], account_id=ctx.account_id, user_id=ctx.user.user_id
            )
            assert task is not None and task.status.value == "failed", task
        else:
            assert outcome.outcome is ProcessOutcome.FAILED, outcome
        records = await fs._get_vector_store().get_strict([record_id], ctx=ctx)
        if change == "record":
            assert records[0]["abstract"] == "intervening record"
            assert records[0]["search_tags"] == ["owner=retain"]
        else:
            assert records == [], records
        assert await fs.read_file_bytes(uri, ctx=ctx) == (
            b"changed source" if change == "source" else original_bytes
        )
        if change not in {"missing", "forged"}:
            assert prepared["resource_index_conditions"] == conditions
        if foreign_handoff is not None:
            owner = await agfs.pathlock_adopt(foreign_handoff)
            assert await agfs.pathlock_refresh(owner) == "refreshed"
            await agfs.pathlock_release(owner)
    finally:
        with suppress(Exception):
            owner = await agfs.pathlock_adopt(original_handoff)
            await agfs.pathlock_release(owner)
