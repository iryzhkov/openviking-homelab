# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Resource SemanticMsg carries the original request's account identity.

The identity is stamped only inside a producer's bound original admission and
processing refuses a missing, deleting, recreated or changed identity before
its first lock or filesystem mutation.
"""

import pytest

from openviking.server.account_stores.models import AccountIdentity
from openviking.storage.queuefs.named_queue import NamedQueue
from openviking.storage.queuefs.process_result import ProcessOutcome
from openviking.storage.queuefs.resource_embedding_fence import (
    ResourceEmbeddingConflict,
    bind_resource_semantic_admission,
)
from openviking.storage.queuefs.semantic_msg import SemanticMsg
from openviking.storage.queuefs.semantic_queue import SemanticQueue
from tests.storage.test_resource_semantic_fence import env as env
from tests.storage.test_resource_semantic_fence import handoff


def _spy_lock_acquisition(env, monkeypatch):
    calls = []
    for name in ("pathlock_acquire_batch", "pathlock_adopt", "pathlock_acquire_tree"):
        original = getattr(env.agfs, name)

        async def spy(*args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            return await _original(*args, **kwargs)

        monkeypatch.setattr(env.agfs, name, spy)
    return calls


@pytest.mark.parametrize(
    "state",
    ["missing", "recreated", "deleting", "absent"],
)
async def test_unadmitted_resource_semantic_work_refuses_before_any_mutation(
    env, monkeypatch, state
):
    kwargs = {}
    if state == "missing":
        kwargs["resource_creation_identity"] = ""
    elif state == "recreated":
        env.fs._account_identity_manager.get_account_identity.return_value = AccountIdentity(
            "recreated-creation", False
        )
    elif state == "deleting":
        env.fs._account_identity_manager.get_account_identity.return_value = AccountIdentity(
            "test-creation", True
        )
    else:
        env.fs._account_identity_manager.get_account_identity.return_value = None
    calls = _spy_lock_acquisition(env, monkeypatch)
    before = sorted(p.name for p in env.target.iterdir())
    result = await env.processor.on_dequeue(env.message(**kwargs).to_dict())
    assert result.outcome is ProcessOutcome.FAILED
    assert calls == []
    env.constructor.assert_not_called()
    assert env.writes == []
    assert env.content.read_bytes() == b"original"
    assert sorted(p.name for p in env.target.iterdir()) == before


async def test_identity_recreated_after_final_coverage_refuses_before_writes(env, monkeypatch):
    original = env.agfs.pathlock_acquire_batch

    async def acquire(*args, **kwargs):
        lease = await original(*args, **kwargs)
        env.fs._account_identity_manager.get_account_identity.return_value = AccountIdentity(
            "recreated-creation", False
        )
        return lease

    monkeypatch.setattr(env.agfs, "pathlock_acquire_batch", acquire)
    result = await env.processor.on_dequeue(env.message().to_dict())
    assert result.outcome is ProcessOutcome.FAILED
    env.constructor.assert_not_called()
    assert env.writes == []
    lease = await env.agfs.pathlock_acquire_tree("/local/account/resources")
    await env.agfs.pathlock_release(lease)


async def test_admitted_resource_semantic_work_runs(env):
    result = await env.processor.on_dequeue(env.message().to_dict())
    assert result.outcome is ProcessOutcome.SUCCESS, result
    env.constructor.assert_called()


def test_message_field_is_strict_and_legacy_compatible():
    msg = SemanticMsg(
        "viking://resources/a", "resource", account_id="acc", resource_creation_identity="c1"
    )
    restored = SemanticMsg.from_dict(msg.to_dict())
    assert restored.resource_creation_identity == "c1"
    legacy = msg.to_dict()
    del legacy["resource_creation_identity"]
    assert SemanticMsg.from_dict(legacy).resource_creation_identity == ""
    bad = msg.to_dict()
    bad["resource_creation_identity"] = 5
    with pytest.raises(ValueError):
        SemanticMsg.from_dict(bad)


@pytest.fixture
def semantic_queue(env, monkeypatch):
    saved = []

    async def save(queue, data):
        del queue
        saved.append(data)
        return "persisted"

    monkeypatch.setattr(NamedQueue, "enqueue", save)
    monkeypatch.setattr("openviking.storage.viking_fs.get_viking_fs", lambda: env.fs)
    return object.__new__(SemanticQueue), saved


async def test_queue_stamps_only_inside_bound_original_admission(env, semantic_queue):
    queue, saved = semantic_queue
    token = await handoff(env)
    await queue.enqueue(env.message(lock_handoff=token, resource_creation_identity=""))
    assert saved[-1]["resource_creation_identity"] == ""
    with bind_resource_semantic_admission("account", "test-creation"):
        await queue.enqueue(env.message(lock_handoff=token, resource_creation_identity=""))
    stamped = SemanticMsg.from_dict(saved[-1])
    assert stamped.resource_creation_identity == "test-creation"
    nonresource = SemanticMsg("viking://user/u/memories", "memory", account_id="account")
    with bind_resource_semantic_admission("account", "test-creation"):
        await queue.enqueue(nonresource, deduplicate_memory=False)
    assert saved[-1]["resource_creation_identity"] == ""


@pytest.mark.parametrize("case", ["no_handoff", "account", "conflict", "recreated", "deleting"])
async def test_queue_refuses_unqualified_admission(env, semantic_queue, case):
    queue, saved = semantic_queue
    token = await handoff(env)
    kwargs = {"lock_handoff": token, "resource_creation_identity": ""}
    account = "account"
    if case == "no_handoff":
        kwargs["lock_handoff"] = None
    elif case == "account":
        account = "other-account"
    elif case == "conflict":
        kwargs["resource_creation_identity"] = "older-creation"
    else:
        env.fs._account_identity_manager.get_account_identity.return_value = AccountIdentity(
            "recreated-creation" if case == "recreated" else "test-creation",
            case == "deleting",
        )
    with bind_resource_semantic_admission(account, "test-creation"):
        with pytest.raises((ResourceEmbeddingConflict, Exception)):
            await queue.enqueue(env.message(**kwargs))
    assert saved == []
