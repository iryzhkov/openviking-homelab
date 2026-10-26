# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Consumer-entry ordering for refused queued resource work.

Both cases deliver a serialized message to the real queue consumer on a real
service: native filesystem and leases, FileStore registry, TaskTracker and
local vector store. Effect recorders always call through.
"""

import asyncio
import json

from openviking.service.task_domain import TaskStatus
from openviking.service.task_tracker import get_task_tracker
from openviking.storage.queuefs.add_resource_msg import AddResourceMsg
from openviking.storage.queuefs.process_result import ProcessOutcome
from tests.server.conftest import sample_markdown_file as sample_markdown_file
from tests.server.conftest import service as service
from tests.server.conftest import temp_dir as temp_dir
from tests.server.test_queued_admission_repair import TARGET, _hold_add_resource, _processor
from tests.server.test_real_resource_producer_e2e import _records, _root_ctx
from tests.server.test_real_resource_producer_e2e import admitted_service as admitted_service


def _record_calls(monkeypatch, owner, name, calls):
    original = getattr(owner, name)

    async def record(*args, **kwargs):
        calls.append(name)
        return await original(*args, **kwargs)

    monkeypatch.setattr(owner, name, record)


async def _sidecars(fs, directory):
    out = {}
    for name in (".abstract.md", ".overview.md"):
        out[name] = await fs._async_agfs.cat(
            fs._uri_to_path(f"{directory}/{name}", ctx=_root_ctx())
        )
    return out


async def test_resource_batch_refresh_refuses_at_consumer_entry(admitted_service, monkeypatch):
    from openviking.storage import content_write as content_write_module
    from openviking.storage.queuefs.content_write_msg import ContentWriteMsg
    from openviking.storage.queuefs.content_write_processor import ContentWriteProcessor

    service = admitted_service
    fs = service.viking_fs
    ctx = _root_ctx()
    directory = "viking://resources/batch-legacy"
    note = directory + "/note.md"
    # Real structured sidecars and vectors from the admitted manual writer.
    await asyncio.wait_for(
        service.fs.write(note, "# Batch legacy\n\nOriginal body.", ctx=ctx, wait=True), timeout=25
    )
    before_sidecars = await _sidecars(fs, directory)
    assert b"freshness:" in before_sidecars[".abstract.md"]
    before_note = await fs._async_agfs.cat(fs._uri_to_path(note, ctx=ctx))
    before_records = sorted(
        json.dumps(r, sort_keys=True, default=str) for r in await _records(service, ctx)
    )

    # A live original handoff: adoption would succeed if the consumer tried.
    lease = await fs._async_agfs.pathlock_acquire_tree(fs._uri_to_path(directory, ctx=ctx))
    handoff = await fs._async_agfs.pathlock_to_handoff(lease)
    await fs._async_agfs.pathlock_handoff(lease)
    tracker = get_task_tracker()
    task = await tracker.create(
        "content_write", account_id=ctx.account_id, user_id=ctx.user.user_id, resource_id=note
    )
    msg = ContentWriteMsg(
        task_id=task.task_id,
        operation="content_write",
        resource_id=note,
        result={"uri": note},
        refresh={
            "kind": "batch",
            "force_refresh": True,
            "refresh_kinds": {note: "modified"},
            "file_md5s": {},
            "file_abstracts": {},
            "sidecar_directories": {},
            "file_ingest_options": {},
            "resource_plans": [],
        },
        account_id=ctx.account_id,
        user_id=ctx.user.user_id,
        role=str(ctx.role),
        lock_handoff=handoff,
    )
    data = json.loads(json.dumps(msg.to_dict()))

    effects = []
    _record_calls(monkeypatch, fs._async_agfs, "pathlock_adopt", effects)
    _record_calls(monkeypatch, fs._async_agfs, "write", effects)
    _record_calls(monkeypatch, fs, "write_file", effects)
    _record_calls(monkeypatch, content_write_module, "plan_abstract_overview_refresh", effects)
    _record_calls(monkeypatch, tracker, "start", effects)

    result = await ContentWriteProcessor(fs).on_dequeue(data)

    assert result.outcome is ProcessOutcome.FAILED, result
    assert "Unbound resource content refresh" in (result.error or ""), result
    assert effects == [], effects
    task_after = await tracker.get(
        task.task_id, account_id=ctx.account_id, user_id=ctx.user.user_id
    )
    assert task_after.status is TaskStatus.FAILED, task_after
    assert "Unbound resource content refresh" in (task_after.error or "")
    # Recorders stay installed (they call through); nothing below is asserted on them.
    assert await _sidecars(fs, directory) == before_sidecars
    assert await fs._async_agfs.cat(fs._uri_to_path(note, ctx=ctx)) == before_note
    assert (
        sorted(json.dumps(r, sort_keys=True, default=str) for r in await _records(service, ctx))
        == before_records
    )
    # The original handoff was never consumed: it is still adoptable.
    adopted = await fs._async_agfs.pathlock_adopt(handoff)
    await fs._async_agfs.pathlock_release(adopted)


async def test_post_process_lost_original_handoff_refuses_before_plan(
    admitted_service, sample_markdown_file, monkeypatch
):
    from openviking.utils.resource_processor import ResourceProcessor

    service = admitted_service
    fs = service.viking_fs
    agfs = fs._async_agfs
    ctx = _root_ctx()
    source = await _hold_add_resource(service, sample_markdown_file, monkeypatch)
    # POST_PROCESS shape of the real producer's persisted message: same task,
    # account, original identity and handoff; prepared plan instead of source.
    payload = dict(source)
    for key in (
        "staged_source",
        "shared_source",
        "understanding_response_id",
        "understanding_file_id",
    ):
        payload.pop(key, None)
    payload["job_phase"] = "post_process"
    payload["prepared"] = {
        "root_uri": TARGET,
        "resource_creation_identity": source["resource_creation_identity"],
        "source_committed": True,
    }
    msg = AddResourceMsg.from_dict(json.loads(json.dumps(payload)))
    assert msg.job_phase.value == "post_process"
    assert msg.prepared and msg.lock_handoff == source["lock_handoff"]
    assert msg.resource_creation_identity == source["resource_creation_identity"]

    # Same lifetime: the original owner settles, another writer leaves a
    # target directory with content.
    adopted = await agfs.pathlock_adopt(source["lock_handoff"])
    await agfs.pathlock_release(adopted)
    target_path = fs._uri_to_path(TARGET, ctx=ctx)
    foreign = await agfs.pathlock_acquire_tree(target_path.rsplit("/", 1)[0])
    try:
        if not await fs.exists(TARGET, ctx=ctx):
            await agfs.mkdir(target_path)
    finally:
        await agfs.pathlock_release(foreign)
    # The foreign writer's own write takes its normal automatic lock.
    await agfs.write(target_path + "/foreign.md", b"foreign writer content")

    reached = []
    _record_calls(monkeypatch, service.resources, "reacquire_add_resource_job_lock", reached)
    _record_calls(monkeypatch, service.resources, "execute_add_resource_job", reached)
    _record_calls(monkeypatch, service.resources, "_cleanup_reserved_target_if_empty", reached)
    _record_calls(monkeypatch, ResourceProcessor, "finish_prepared_resource", reached)
    payload = msg.to_dict()
    payload["lock_handoff_retry"] = 2
    result = await _processor(service).on_dequeue({"data": json.dumps(payload)})

    assert result.outcome is ProcessOutcome.FAILED, result
    assert "lock_handoff" in (result.error or ""), result
    assert reached == [], reached
    assert await agfs.cat(target_path + "/foreign.md") == b"foreign writer content"
    task = await get_task_tracker().get(
        msg.task_id, account_id=ctx.account_id, user_id=ctx.user.user_id
    )
    assert task is not None and task.status is TaskStatus.FAILED, task
    lease = await agfs.pathlock_acquire_tree(target_path)
    await agfs.pathlock_release(lease)
