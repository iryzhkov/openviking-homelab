import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

# Import before per-test dependency patches. A lazy import under a patched
# get_service would cache that fake inside reindex_executor for later tests.
from openviking.service.reindex_executor import ReindexExecutor
from openviking.service.task_tracker import TaskStatus
from openviking.storage.errors import LockAcquisitionError
from openviking.storage.queuefs.reindex_msg import ReindexMsg


def test_reindex_msg_round_trips_only_request_descriptor():
    message = ReindexMsg(
        task_id="task-1",
        uri="viking://resources/demo",
        object_type="resource",
        mode="semantic_and_vectors",
        force=True,
        recursive=False,
        tags=["team=search"],
        tag_mode="append",
        account_id="account-1",
        user_id="user-1",
        group_ids=["group-1"],
        role="admin",
        actor_peer_id="peer-1",
        bypass_acl=True,
        lock_handoff={"lease_ref": "lease-1"},
        resource_creation_identity="original-request-identity",
    )

    restored = ReindexMsg.from_dict(message.to_dict())

    assert restored == message
    assert "source_contents" not in restored.to_dict()
    assert "snapshot" not in restored.to_dict()


def test_reindex_processor_restores_actor_peer_and_acl_bypass_context():
    from openviking.storage.queuefs.reindex_processor import ReindexProcessor

    message = ReindexMsg(
        task_id="task-1",
        uri="viking://resources/demo",
        object_type="resource",
        mode="vectors_only",
        account_id="account-1",
        user_id="user-1",
        role="admin",
        actor_peer_id="peer-1",
        bypass_acl=True,
    )

    ctx = ReindexProcessor._ctx(message)

    assert ctx.actor_peer_id == "peer-1"
    assert ctx.bypass_acl is True


def test_reindex_msg_rejects_missing_descriptor_fields():
    with pytest.raises(ValueError, match="task_id"):
        ReindexMsg.from_dict({"uri": "viking://resources/demo"})


# These doubles qualify only processor/task plumbing. Native admission and
# ownership are covered separately by the real caller integration cases.
_ADMITTED_CREATION_ID = "unit-original-resource-creation"


def _admitted_resource_fs(agfs):
    from openviking.server.account_stores.models import AccountIdentity

    return SimpleNamespace(
        _async_agfs=agfs,
        _account_identity_manager=SimpleNamespace(
            get_account_identity=AsyncMock(
                return_value=AccountIdentity(_ADMITTED_CREATION_ID, False)
            )
        ),
    )


def _admitted_resource_message(**kwargs):
    kwargs.setdefault("lock_handoff", {"lease_ref": "original-request-handoff"})
    return ReindexMsg(resource_creation_identity=_ADMITTED_CREATION_ID, **kwargs)


@pytest.mark.asyncio
async def test_reindex_processor_terminalizes_root_after_descendants(monkeypatch):
    from openviking.service.task_store import PersistentTaskStore, WorkRegistration
    from openviking.service.task_tracker import TaskTracker
    from openviking.storage.queuefs.reindex_processor import ReindexProcessor
    from tests.test_task_tracker import _FakeAgfs

    owner = {"account_id": "account-1", "user_id": "user-1"}
    tracker = TaskTracker(PersistentTaskStore(_FakeAgfs()))
    monkeypatch.setattr(
        "openviking.storage.queuefs.reindex_processor.get_task_tracker", Mock(return_value=tracker)
    )
    monkeypatch.setattr(
        "openviking.server.dependencies.get_service",
        lambda: SimpleNamespace(_vlm_resolver=None, _vector_config_resolver=None),
    )

    result = {"status": "completed", "rebuilt_records": 3}
    monkeypatch.setattr(ReindexExecutor, "_run", AsyncMock(return_value=result))
    viking_fs = _admitted_resource_fs(SimpleNamespace(pathlock_release=AsyncMock()))
    processor = ReindexProcessor(viking_fs)
    processor._adopt_or_reacquire = AsyncMock(return_value={"lease": "root"})
    task = await tracker.create("admin_reindex", resource_id="viking://resources/demo", **owner)
    assert (
        await tracker.register_work(task.task_id, "root-work", "Reindex", **owner)
        is WorkRegistration.CREATED
    )
    assert (
        await tracker.register_work(task.task_id, "child-work", "Embedding", **owner)
        is WorkRegistration.CREATED
    )
    entered_descendant_wait = asyncio.Event()
    original_wait_for_task_work = tracker.wait_for_task_work

    async def wait_for_task_work(*args, **kwargs):
        entered_descendant_wait.set()
        await original_wait_for_task_work(*args, **kwargs)

    monkeypatch.setattr(tracker, "wait_for_task_work", wait_for_task_work)
    message = _admitted_resource_message(
        task_id=task.task_id,
        uri="viking://resources/demo",
        object_type="resource",
        mode="vectors_only",
        **owner,
        role="admin",
    )

    processing = asyncio.create_task(
        processor.on_dequeue(
            {"id": "queue-1", "data": {**message.to_dict(), "_task_work_id": "root-work"}}
        )
    )
    await asyncio.wait_for(entered_descendant_wait.wait(), 1)
    assert not processing.done()
    assert (await tracker.get(task.task_id, **owner)).status is TaskStatus.RUNNING

    await tracker.mark_work_done(task.task_id, "child-work", **owner)
    outcome = await asyncio.wait_for(processing, 1)
    assert outcome.outcome.value == "success"
    assert (await tracker.get(task.task_id, **owner)).status is TaskStatus.RUNNING

    await tracker.mark_work_done(task.task_id, "root-work", **owner)
    completed = await tracker.wait(task.task_id, **owner, timeout=0)
    assert completed.status is TaskStatus.COMPLETED
    assert completed.result["rebuilt_records"] == 3
    assert completed.result["queue_status"]["Embedding"]["processed"] == 1


@pytest.mark.asyncio
async def test_reindex_processor_restores_root_telemetry_for_executor(monkeypatch):
    from openviking.storage.queuefs.reindex_processor import ReindexProcessor
    from openviking.telemetry import (
        OperationTelemetry,
        get_current_telemetry,
        register_telemetry,
        unregister_telemetry,
    )

    tracker = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(status=TaskStatus.RUNNING)),
        start=AsyncMock(),
        complete=AsyncMock(),
        fail=AsyncMock(),
        wait_for_task_work=AsyncMock(),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.reindex_processor.get_task_tracker", Mock(return_value=tracker)
    )
    monkeypatch.setattr(
        "openviking.server.dependencies.get_service",
        lambda: SimpleNamespace(_vlm_resolver=None, _vector_config_resolver=None),
    )
    telemetry = OperationTelemetry(operation="content_reindex", enabled=True)
    telemetry_id = telemetry.telemetry_id
    register_telemetry(telemetry)
    seen = {}

    async def capture_telemetry(_self, **_kwargs):
        seen["telemetry_id"] = get_current_telemetry().telemetry_id
        return {"status": "completed"}

    monkeypatch.setattr(
        ReindexExecutor,
        "_run",
        capture_telemetry,
    )
    processor = ReindexProcessor(
        _admitted_resource_fs(SimpleNamespace(pathlock_release=AsyncMock()))
    )
    processor._adopt_or_reacquire = AsyncMock(return_value={"lease": "root"})
    message = _admitted_resource_message(
        task_id="task-1",
        uri="viking://resources/demo",
        object_type="resource",
        mode="semantic_and_vectors",
        account_id="account-1",
        user_id="user-1",
        role="admin",
        telemetry_id=telemetry_id,
    )

    try:
        outcome = await processor.on_dequeue({"id": "queue-1", "data": message.to_dict()})
    finally:
        unregister_telemetry(telemetry_id)

    assert outcome.outcome.value == "success"
    assert seen["telemetry_id"] == telemetry_id


@pytest.mark.asyncio
async def test_reindex_processor_replay_of_completed_task_skips_force_rebuild(monkeypatch):
    from openviking.storage.queuefs.reindex_processor import ReindexProcessor

    tracker = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(status=TaskStatus.COMPLETED)),
        start=AsyncMock(),
        complete=AsyncMock(),
        fail=AsyncMock(),
        wait_for_task_work=AsyncMock(),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.reindex_processor.get_task_tracker", Mock(return_value=tracker)
    )
    monkeypatch.setattr(
        "openviking.server.dependencies.get_service",
        lambda: SimpleNamespace(_vlm_resolver=None, _vector_config_resolver=None),
    )
    monkeypatch.setattr(
        ReindexExecutor,
        "_run",
        AsyncMock(return_value={"status": "completed"}),
    )
    viking_fs = _admitted_resource_fs(SimpleNamespace(pathlock_release=AsyncMock()))
    processor = ReindexProcessor(viking_fs)
    processor._adopt_or_reacquire = AsyncMock(return_value={"lease": "must-not-adopt"})
    message = _admitted_resource_message(
        task_id="task-1",
        uri="viking://resources/demo",
        object_type="resource",
        mode="semantic_and_vectors",
        force=True,
        account_id="account-1",
        user_id="user-1",
        role="admin",
        lock_handoff={"lease_ref": "old"},
    )

    outcome = await processor.on_dequeue(
        {"id": "queue-1", "data": {**message.to_dict(), "_task_work_id": "root-work"}}
    )

    assert outcome.outcome.value == "success"
    processor._adopt_or_reacquire.assert_not_awaited()
    tracker.start.assert_not_awaited()
    tracker.complete.assert_not_awaited()
    tracker.fail.assert_not_awaited()


@pytest.mark.asyncio
async def test_reindex_processor_requeues_lock_handoff_failure_before_ack(monkeypatch):
    from openviking.storage.queuefs.reindex_processor import ReindexProcessor

    tracker = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(status=TaskStatus.RUNNING)),
        start=AsyncMock(),
        complete=AsyncMock(),
        fail=AsyncMock(),
        wait_for_task_work=AsyncMock(),
    )
    queue_manager = SimpleNamespace(REINDEX="Reindex", enqueue=AsyncMock(return_value="retry-1"))
    monkeypatch.setattr(
        "openviking.storage.queuefs.reindex_processor.get_task_tracker", Mock(return_value=tracker)
    )
    monkeypatch.setattr("openviking.storage.queuefs.get_queue_manager", lambda: queue_manager)
    # A temporary delivery/adoption error can retry the SAME original handoff.
    # It grants no authority to reacquire coverage or replace the identity.
    agfs = SimpleNamespace(
        pathlock_adopt=AsyncMock(side_effect=LockAcquisitionError("handoff delivery unavailable")),
        pathlock_acquire_tree=AsyncMock(),
        pathlock_acquire_exact=AsyncMock(),
        pathlock_release=AsyncMock(),
    )
    viking_fs = _admitted_resource_fs(agfs)
    viking_fs.stat = AsyncMock()
    processor = ReindexProcessor(viking_fs)
    message = _admitted_resource_message(
        task_id="task-1",
        uri="viking://resources/demo",
        object_type="resource",
        mode="vectors_only",
        account_id="account-1",
        user_id="user-1",
        role="admin",
    )

    outcome = await processor.on_dequeue(
        {"id": "queue-1", "data": {**message.to_dict(), "_task_work_id": "root-work"}}
    )

    assert outcome.outcome.value == "requeued"
    queue_manager.enqueue.assert_awaited_once_with(
        "Reindex",
        {
            **message.to_dict(),
            "_task_work_id": "root-work",
            "lock_handoff_retry": 1,
        },
    )
    tracker.fail.assert_not_awaited()
    agfs.pathlock_adopt.assert_awaited_once_with(message.lock_handoff)
    agfs.pathlock_acquire_tree.assert_not_awaited()
    agfs.pathlock_acquire_exact.assert_not_awaited()
    agfs.pathlock_release.assert_not_awaited()
    viking_fs.stat.assert_not_awaited()


@pytest.mark.asyncio
async def test_reindex_processor_fails_after_lock_handoff_retry_limit(monkeypatch):
    from openviking.storage.queuefs.reindex_processor import ReindexProcessor

    tracker = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(status=TaskStatus.RUNNING)),
        start=AsyncMock(),
        complete=AsyncMock(),
        fail=AsyncMock(),
        wait_for_task_work=AsyncMock(),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.reindex_processor.get_task_tracker", Mock(return_value=tracker)
    )
    processor = ReindexProcessor(_admitted_resource_fs(SimpleNamespace()))
    processor._adopt_or_reacquire = AsyncMock(side_effect=LockAcquisitionError("busy"))
    message = _admitted_resource_message(
        task_id="task-1",
        uri="viking://resources/demo",
        object_type="resource",
        mode="vectors_only",
        account_id="account-1",
        user_id="user-1",
        role="admin",
        lock_handoff_retry=2,
    )

    outcome = await processor.on_dequeue({"id": "queue-1", "data": message.to_dict()})

    assert outcome.outcome.value == "failed"
    assert "Invalid lock_handoff" in outcome.error
    assert "busy" in outcome.error
    processor._adopt_or_reacquire.assert_awaited_once()
    tracker.fail.assert_awaited_once()


@pytest.mark.asyncio
async def test_reindex_processor_cancellation_releases_root_lock_handoff():
    from openviking.storage.queuefs.reindex_processor import ReindexProcessor

    agfs = SimpleNamespace(
        pathlock_adopt=AsyncMock(return_value={"lease": "root"}),
        pathlock_release=AsyncMock(),
    )
    processor = ReindexProcessor(_admitted_resource_fs(agfs))
    message = _admitted_resource_message(
        task_id="task-1",
        uri="viking://resources/demo",
        object_type="resource",
        mode="vectors_only",
        account_id="account-1",
        user_id="user-1",
        role="admin",
        lock_handoff={"lease_ref": "root-handoff"},
    )

    outcome = await processor.on_cancelled({"data": message.to_dict()})

    assert outcome.outcome.value == "cancelled"
    agfs.pathlock_adopt.assert_awaited_once_with({"lease_ref": "root-handoff"})
    agfs.pathlock_release.assert_awaited_once_with({"lease": "root"})


@pytest.mark.asyncio
async def test_reindex_processor_does_not_release_lease_owned_by_failed_executor(monkeypatch):
    from openviking.storage.queuefs.reindex_processor import ReindexProcessor

    tracker = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(status=TaskStatus.RUNNING)),
        start=AsyncMock(),
        complete=AsyncMock(),
        fail=AsyncMock(),
        wait_for_task_work=AsyncMock(),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.reindex_processor.get_task_tracker", Mock(return_value=tracker)
    )
    monkeypatch.setattr(
        "openviking.server.dependencies.get_service",
        lambda: SimpleNamespace(_vlm_resolver=None, _vector_config_resolver=None),
    )
    agfs = SimpleNamespace(pathlock_release=AsyncMock())

    async def fail_after_releasing(_self, *, existing_lease, **_kwargs):
        await agfs.pathlock_release(existing_lease)
        raise RuntimeError("semantic plan enqueue failed")

    monkeypatch.setattr(
        ReindexExecutor,
        "_run",
        fail_after_releasing,
    )
    processor = ReindexProcessor(_admitted_resource_fs(agfs))
    processor._adopt_or_reacquire = AsyncMock(return_value={"lease": "root"})
    message = _admitted_resource_message(
        task_id="task-1",
        uri="viking://resources/demo",
        object_type="resource",
        mode="semantic_and_vectors",
        account_id="account-1",
        user_id="user-1",
        role="admin",
    )

    outcome = await processor.on_dequeue({"id": "queue-1", "data": message.to_dict()})

    assert outcome.outcome.value == "failed"
    agfs.pathlock_release.assert_awaited_once_with({"lease": "root"})
    tracker.fail.assert_awaited_once()


@pytest.mark.parametrize("legacy_identity", ["", None, 123])
def test_reindex_msg_identity_is_strict_and_legacy_default_is_unbound(legacy_identity):
    message = _admitted_resource_message(
        task_id="task-1",
        uri="viking://resources/demo",
        object_type="resource",
        mode="vectors_only",
        account_id="account-1",
        user_id="user-1",
        role="admin",
    )
    payload = message.to_dict()
    payload.pop("resource_creation_identity")
    assert ReindexMsg.from_dict(payload).resource_creation_identity == ""
    payload["resource_creation_identity"] = legacy_identity
    if isinstance(legacy_identity, str):
        assert ReindexMsg.from_dict(payload).resource_creation_identity == ""
    else:
        with pytest.raises(ValueError, match="string"):
            ReindexMsg.from_dict(payload)


@pytest.mark.asyncio
async def test_legacy_resource_job_refuses_before_capture_adoption_or_native_writes(monkeypatch):
    from openviking.storage.queuefs.reindex_processor import ReindexProcessor

    tracker = SimpleNamespace(fail=AsyncMock(), get=AsyncMock())
    monkeypatch.setattr(
        "openviking.storage.queuefs.reindex_processor.get_task_tracker",
        lambda: tracker,
    )
    agfs = SimpleNamespace(
        pathlock_adopt=AsyncMock(),
        pathlock_acquire_tree=AsyncMock(),
        pathlock_acquire_exact=AsyncMock(),
        pathlock_release=AsyncMock(),
        write=AsyncMock(),
        mkdir=AsyncMock(),
    )
    fs = _admitted_resource_fs(agfs)
    fs._capture_resource_identity = AsyncMock(side_effect=AssertionError("must not recapture"))
    fs.stat = AsyncMock()
    message = ReindexMsg(
        task_id="task-legacy",
        uri="viking://resources/demo",
        object_type="resource",
        mode="vectors_only",
        account_id="account-1",
        user_id="user-1",
        role="admin",
        lock_handoff={"lease_ref": "legacy-handoff"},
    )

    outcome = await ReindexProcessor(fs).on_dequeue({"data": message.to_dict()})

    assert outcome.outcome.value == "failed"
    assert "original request admission" in outcome.error
    tracker.fail.assert_awaited_once()
    tracker.get.assert_not_awaited()
    fs._capture_resource_identity.assert_not_awaited()
    fs._account_identity_manager.get_account_identity.assert_not_awaited()
    fs.stat.assert_not_awaited()
    for method in vars(agfs).values():
        method.assert_not_awaited()


@pytest.mark.asyncio
async def test_resource_retry_rechecks_original_identity_before_adoption(monkeypatch):
    from openviking.server.account_stores.models import AccountIdentity
    from openviking.storage.queuefs.reindex_processor import ReindexProcessor

    tracker = SimpleNamespace(fail=AsyncMock(), get=AsyncMock())
    queue_manager = SimpleNamespace(REINDEX="Reindex", enqueue=AsyncMock())
    monkeypatch.setattr(
        "openviking.storage.queuefs.reindex_processor.get_task_tracker", lambda: tracker
    )
    monkeypatch.setattr("openviking.storage.queuefs.get_queue_manager", lambda: queue_manager)
    agfs = SimpleNamespace(pathlock_adopt=AsyncMock(), pathlock_release=AsyncMock())
    fs = _admitted_resource_fs(agfs)
    fs._account_identity_manager.get_account_identity.return_value = AccountIdentity(
        "replacement-account-identity", False
    )
    message = _admitted_resource_message(
        task_id="task-retry",
        uri="viking://resources/demo",
        object_type="resource",
        mode="vectors_only",
        account_id="account-1",
        user_id="user-1",
        role="admin",
        lock_handoff_retry=1,
    )

    outcome = await ReindexProcessor(fs).on_dequeue({"data": message.to_dict()})

    assert outcome.outcome.value == "failed"
    assert "lifetime changed" in outcome.error
    agfs.pathlock_adopt.assert_not_awaited()
    agfs.pathlock_release.assert_not_awaited()
    queue_manager.enqueue.assert_not_awaited()
    tracker.fail.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_stale_resource_handoff_never_releases_foreign_owner():
    from openviking.storage.queuefs.reindex_processor import ReindexProcessor

    agfs = SimpleNamespace(
        pathlock_adopt=AsyncMock(side_effect=LockAcquisitionError("original handoff lost")),
        pathlock_release=AsyncMock(),
        pathlock_acquire_tree=AsyncMock(),
        pathlock_acquire_exact=AsyncMock(),
    )
    message = _admitted_resource_message(
        task_id="task-cancelled",
        uri="viking://resources/demo",
        object_type="resource",
        mode="vectors_only",
        account_id="account-1",
        user_id="user-1",
        role="admin",
    )

    outcome = await ReindexProcessor(_admitted_resource_fs(agfs)).on_cancelled(
        {"data": message.to_dict()}
    )

    assert outcome.outcome.value == "cancelled"
    agfs.pathlock_adopt.assert_awaited_once_with(message.lock_handoff)
    agfs.pathlock_release.assert_not_awaited()
    agfs.pathlock_acquire_tree.assert_not_awaited()
    agfs.pathlock_acquire_exact.assert_not_awaited()
