"""Original file-parent admission through the real SDK and native queued work."""

import asyncio
import hashlib
import inspect
import json
import os
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import uvicorn

from openviking.server.auth import get_request_context
from openviking.server.identity import RequestContext, Role
from openviking.storage.acl import AclMode
from openviking.storage.queuefs.add_resource_processor import AddResourceProcessor
from openviking.storage.queuefs.named_queue import NamedQueue
from openviking.storage.queuefs.semantic_work import ResourceSemanticMessageWork
from openviking.utils.resource_processor import ResourceProcessor
from openviking.utils.summarizer import Summarizer
from openviking_cli.exceptions import PermissionDeniedError
from openviking_cli.session.user_id import UserIdentifier
from tests.server.conftest import app as app
from tests.server.conftest import service as service
from tests.server.conftest import temp_dir as temp_dir
from tests.server.test_real_resource_producer_e2e import _root_ctx
from tests.server.test_real_resource_producer_e2e import admitted_service as admitted_service

ROOT = "viking://resources/composite"
RAW = ROOT + "/raw.md"


def owner(value):
    if value is None:
        return None
    observed = {key: value.get(key) for key in ("owner_id", "lease_ref", "lock_paths")}
    capability = value.get("ownership_ref")
    if isinstance(capability, str) and capability:
        observed.update(
            capability_state="observed",
            capability_sha256=hashlib.sha256(capability.encode()).hexdigest(),
        )
    else:
        observed["capability_state"] = "not_observed"
    return observed


@asynccontextmanager
async def backend_tcp(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    server = uvicorn.Server(
        uvicorn.Config(app, lifespan="off", log_level="critical", access_log=False)
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield sock.getsockname()[1]
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)
        sock.close()


async def test_sdk_raw_parent_original_admission(admitted_service, app, monkeypatch):
    svc, ctx = admitted_service, _root_ctx()
    fs, out = svc.viking_fs, Path(os.environ["PROOF_OUT"])
    proof = {"events": [], "body_denials": [], "teardown": "pending"}

    def record(event, **fields):
        proof["events"].append({"event": event, **fields})
        (out / "producer-proof.json").write_text(json.dumps(proof, indent=2) + "\n")

    app.dependency_overrides[get_request_context] = lambda: ctx
    await fs.mkdir(ROOT, ctx=ctx)
    await fs.mkdir(ROOT + "/archive", ctx=ctx)
    await fs.write_file_bytes(RAW, b"old raw\n", ctx=ctx)
    await fs._get_vector_store().upsert(
        {
            "id": RAW,
            "uri": RAW,
            "level": 2,
            "account_id": ctx.account_id,
            "abstract": "old",
            "vector": [0.1] * 2048,
            "search_tags": ["owner=retained"],
        },
        ctx=ctx,
    )
    await svc.runtime_config_manager.patch_account(ctx.account_id, {"acl": {"enabled": True}})
    await fs.set_acl(
        RAW,
        [{"principal": "user:" + ctx.user.user_id, "level": "manage"}],
        acl_mode=AclMode.RESTRICTED,
        ctx=ctx,
    )
    outsider = RequestContext(user=UserIdentifier(ctx.account_id, "outsider"), role=Role.USER)
    write = fs.write_file_bytes

    async def protected_write(uri, data, *args, **kwargs):
        result = await write(uri, data, *args, **kwargs)
        if uri == RAW or uri.startswith(ROOT + "/archive/raw.md.backup-"):
            with pytest.raises(PermissionDeniedError):
                await fs.read_file_bytes(uri, ctx=outsider)
            proof["body_denials"].append(uri)
            record("protected_body", uri=uri, sha256=hashlib.sha256(data).hexdigest())
        return result

    monkeypatch.setattr(fs, "write_file_bytes", protected_write)
    original_adopt = AddResourceProcessor._load_lock

    async def adopt(self, msg, context):
        record("add_input", identity=msg.resource_creation_identity, owner=owner(msg.lock_handoff))
        result = await original_adopt(self, msg, context)
        record("add_adopted", owner=owner(result))
        return result

    monkeypatch.setattr(AddResourceProcessor, "_load_lock", adopt)
    finish = ResourceProcessor.finish_prepared_resource

    async def finish_prepared(self, prepared, **kwargs):
        from openviking.storage.context_update_plan import ContextUpdatePlan

        plan = ContextUpdatePlan.from_dict(prepared["context_update_plan"])
        actions = plan.direct_index_actions
        capture = self._prepared_index_owner(
            prepared, kwargs["ctx"], execution_owner=kwargs["resource_lock"]
        )
        conditions = prepared.get("resource_index_conditions")
        assert isinstance(conditions, dict)
        if actions:
            assert capture is not None
            assert all(
                f"{action.action.value}:{action.record_id}" in conditions for action in actions
            )
        else:
            assert capture is None and not conditions
        record(
            "finish_enter",
            identity=prepared["resource_creation_identity"],
            owner=owner(kwargs["resource_lock"]),
            direct_index_action_count=len(actions),
            capture=owner(capture),
            capture_conditions={
                key: {
                    **{
                        field: value.get(field)
                        for field in (
                            "account_id",
                            "creation_identity",
                            "uri",
                            "level",
                            "record_id",
                            "action",
                            "context_sha256",
                            "input_sha256",
                            "input_state",
                            "intent_sha256",
                        )
                        if field in value
                    },
                    "owner": owner(value.get("owner")),
                }
                for key, value in conditions.items()
            },
        )
        result = await finish(self, prepared, **kwargs)
        record("finish_return")
        return result

    monkeypatch.setattr(ResourceProcessor, "finish_prepared_resource", finish_prepared)
    for method in ("_capture_resource_identity", "_acquire_resource_coverage"):
        original = getattr(fs, method)

        async def observe_authority(*args, _original=original, _method=method, **kwargs):
            caller = inspect.currentframe().f_back.f_code
            record(
                "authority_call", method=_method, caller=caller.co_filename, function=caller.co_name
            )
            return await _original(*args, **kwargs)

        monkeypatch.setattr(fs, method, observe_authority)
    refresh = Summarizer.refresh_file_parent

    async def refresh_parent(self, **kwargs):
        record("parent_producer", uri=kwargs["file_uri"], owner=owner(kwargs.get("lock_handoff")))
        return await refresh(self, **kwargs)

    monkeypatch.setattr(Summarizer, "refresh_file_parent", refresh_parent)
    enqueue = NamedQueue.enqueue

    async def persisted(self, data):
        if isinstance(data, dict) and data.get("uri") == ROOT and "generation_trigger" in data:
            record(
                "parent_queued",
                uri=data["uri"],
                changes=data.get("changes"),
                identity=data.get("resource_creation_identity"),
                owner=owner(data.get("lock_handoff")),
                coalesce_key=data.get("coalesce_key"),
            )
        return await enqueue(self, data)

    monkeypatch.setattr(NamedQueue, "enqueue", persisted)
    acquire = ResourceSemanticMessageWork.acquire_lock

    async def acquire_semantic(self, context):
        if self.msg.uri == ROOT:
            record(
                "semantic_input",
                identity=self.msg.resource_creation_identity,
                owner=owner(self.msg.lock_handoff),
                uri=self.msg.uri,
            )
        result = await acquire(self, context)
        if self.msg.uri == ROOT:
            assert await fs._async_agfs.pathlock_refresh(self.scope.lock) == "refreshed"
            record("semantic_adopted", owner=owner(self.scope.lock))
        return result

    monkeypatch.setattr(ResourceSemanticMessageWork, "acquire_lock", acquire_semantic)
    finish_semantic = ResourceSemanticMessageWork.finish_processing

    async def semantic_finished(self, succeeded):
        if self.msg.uri == ROOT:
            record("semantic_finish_start", succeeded=succeeded)
        result = await finish_semantic(self, succeeded)
        if self.msg.uri == ROOT:
            record("semantic_finish_done", succeeded=succeeded)
        return result

    monkeypatch.setattr(ResourceSemanticMessageWork, "finish_processing", semantic_finished)
    release = fs._async_agfs.pathlock_release

    async def released(lease):
        record("release_attempt", owner=owner(lease))
        return await release(lease)

    monkeypatch.setattr(fs._async_agfs, "pathlock_release", released)
    close = svc.close

    async def closed():
        try:
            await close()
        except BaseException:
            proof["teardown"] = "failed"
            record("teardown_failed")
            raise
        proof["teardown"] = "passed"
        record("teardown_passed")

    monkeypatch.setattr(svc, "close", closed)
    script = Path(__file__).with_name("document_file_parent_sdk.py")
    env = dict(
        os.environ,
        PYTHONPATH=os.environ["COMPOSITE_SDK_PATH"],
        COMPOSITE_HTTP_SOURCE=str(Path(os.environ["PROOF_SOURCE"]) / "http-input"),
        COMPOSITE_RESULT=str(out / "sdk-proof.json"),
    )
    async with backend_tcp(app) as port:
        env["COMPOSITE_BACKEND"] = f"http://127.0.0.1:{port}"
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            log, _ = await asyncio.wait_for(child.communicate(), 75)
        finally:
            if child.returncode is None:
                child.kill()
                await child.wait()
        (out / "sdk.log").write_bytes(log)
        assert child.returncode == 0, log.decode()[-12000:]
    sdk = json.loads((out / "sdk-proof.json").read_text())
    ack = sdk["ack"]
    assert ack["uri"] == RAW and ack["representation"] == "raw_file"
    assert await fs.read_file_bytes(RAW, ctx=ctx) == b"standard raw cobalt"
    assert await fs.read_file_bytes(ack["backup"]["uri"], ctx=ctx) == b"old raw\n"
    assert set(proof["body_denials"]) == {RAW, ack["backup"]["uri"]}
    queues = await asyncio.wait_for(svc.resources.wait_processed(), 35)
    proof["queues"] = queues
    record("queues_settled")
    assert all(status["error_count"] == 0 for status in queues.values()), queues
    assert queues["Semantic"]["processed"] >= 1
    events = proof["events"]

    def one(name):
        rows = [event for event in events if event["event"] == name]
        assert len(rows) == 1, (name, rows)
        return rows[0]

    original, execution = one("add_input"), one("finish_enter")
    producer, queued, consumer = one("parent_producer"), one("parent_queued"), one("semantic_input")
    adopted = one("semantic_adopted")
    assert (
        original["identity"] == execution["identity"] == queued["identity"] == consumer["identity"]
    )
    assert queued["changes"] == {"modified": [RAW]} and queued["uri"] == ROOT
    assert not queued["coalesce_key"]
    owners = [
        original["owner"],
        one("add_adopted")["owner"],
        execution["owner"],
        producer["owner"],
        queued["owner"],
        consumer["owner"],
        adopted["owner"],
    ]
    if execution["direct_index_action_count"]:
        assert execution["capture"] is not None and execution["capture_conditions"]
        owners.append(execution["capture"])
    else:
        assert execution["capture"] is None and execution["capture_conditions"] == {}
    assert all(value is not None for value in owners)
    assert all(value["owner_id"] == owners[0]["owner_id"] for value in owners)
    assert all(value["lock_paths"] == owners[0]["lock_paths"] for value in owners)
    assert one("add_adopted")["owner"] == execution["owner"]
    assert (
        producer["owner"]["lease_ref"]
        == queued["owner"]["lease_ref"]
        == execution["owner"]["lease_ref"]
    )
    assert consumer["owner"] == queued["owner"]
    for value in owners:
        if value["capability_state"] == "not_observed":
            assert "capability_sha256" not in value
    assert (
        execution["owner"]["capability_state"] == adopted["owner"]["capability_state"] == "observed"
    )
    assert execution["owner"]["capability_sha256"] != adopted["owner"]["capability_sha256"]
    stop = events.index(one("semantic_finish_done"))
    assert not any(
        event["event"] == "release_attempt" and event["owner"] == execution["owner"]
        for event in events[events.index(execution) : stop]
    )
    assert one("semantic_finish_done")["succeeded"] is True
    producer_releases = [
        index
        for index, event in enumerate(events)
        if event["event"] == "release_attempt" and event["owner"] == execution["owner"]
    ]
    assert len(producer_releases) == 1 and producer_releases[0] > stop
    consumer_releases = [
        index
        for index, event in enumerate(events)
        if event["event"] == "release_attempt" and event["owner"] == adopted["owner"]
    ]
    assert len(consumer_releases) == 1
    assert events.index(one("semantic_finish_start")) < consumer_releases[0] < stop
    assert not any(
        event["event"] == "authority_call"
        and Path(event["caller"]).name in {"resource_processor.py", "summarizer.py"}
        for event in events[events.index(execution) : events.index(one("finish_return"))]
    )
    proof["parent_bodies"] = {}
    for name in (".abstract.md", ".overview.md"):
        raw = await fs.read_file_bytes(ROOT + "/" + name, ctx=ctx)
        assert raw
        proof["parent_bodies"][name] = hashlib.sha256(raw).hexdigest()
    proof["state"] = "passed"
    record("native_parent_refresh_verified")


@pytest.mark.parametrize("failure", ["lost", "cancelled"])
async def test_file_parent_refuses_before_handoff(admitted_service, monkeypatch, failure):
    svc, ctx = admitted_service, _root_ctx()
    fs, processor = svc.viking_fs, svc.resources._resource_processor
    await fs.mkdir(ROOT, ctx=ctx)
    await fs.write_file_bytes(RAW, b"unchanged", ctx=ctx)
    admitted = await fs._capture_resource_identity(ctx)
    lease = await fs._acquire_resource_coverage(ctx, admitted)
    prepared = {
        "root_uri": RAW,
        "source_committed": True,
        "root_is_file": True,
        "resource_creation_identity": admitted.creation_identity,
        "target_preexisting": True,
    }
    events = []
    summarizer = processor._summarizer_for

    async def interrupt(context):
        result = await summarizer(context)
        if failure == "lost":
            await fs._async_agfs.pathlock_release(lease)
            events.append("original_released")
            return result
        events.append("cancelled_before_handoff")
        raise asyncio.CancelledError()

    monkeypatch.setattr(processor, "_summarizer_for", interrupt)
    enqueue = NamedQueue.enqueue

    async def record_queue(self, data):
        if isinstance(data, dict) and data.get("uri") == ROOT and "generation_trigger" in data:
            events.append("parent_enqueued")
        return await enqueue(self, data)

    monkeypatch.setattr(NamedQueue, "enqueue", record_queue)

    async def no_fresh_authority(*args, **kwargs):
        events.append("fresh_authority")
        raise AssertionError("must not reacquire or recapture after admission")

    monkeypatch.setattr(fs, "_capture_resource_identity", no_fresh_authority)
    monkeypatch.setattr(fs, "_acquire_resource_coverage", no_fresh_authority)
    expected = asyncio.CancelledError if failure == "cancelled" else Exception
    with pytest.raises(expected):
        await processor.finish_prepared_resource(
            prepared, ctx=ctx, resource_lock=lease, build_index=True
        )
    assert events == ["original_released" if failure == "lost" else "cancelled_before_handoff"]
    assert await fs.read_file_bytes(RAW, ctx=ctx) == b"unchanged"
    Path(os.environ["PROOF_OUT"], "failure-proof.json").write_text(
        json.dumps(
            {
                "failure": failure,
                "events": events,
                "state": "refused_before_handoff",
                "original_owner": owner(lease),
            },
            indent=2,
        )
        + "\n"
    )
