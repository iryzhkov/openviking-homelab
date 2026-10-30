"""USER resource maintenance reaches the real route and fenced native executor."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openviking_cli.exceptions import PermissionDeniedError

from openviking.server.auth import get_request_context
from openviking.server.identity import RequestContext, Role
from openviking.server.routers.content import ReindexRequest, reindex
from openviking.storage.acl import AclAction, AclMode
from tests.server import conftest as fixtures
from tests.server import test_vectors_only_reindex_admission_e2e as vector_fixtures
from tests.server.test_vectors_only_reindex_admission_e2e import (
    _forbid_models,
    _root_ctx,
    _seed,
)

admitted_service = vector_fixtures.admitted_service
persisted = vector_fixtures.persisted

app = fixtures.app
client = fixtures.client
service = fixtures.service
temp_dir = fixtures.temp_dir


@pytest.mark.parametrize("allowed", [False, True])
async def test_user_resource_route_checks_manage_before_work(monkeypatch, allowed):
    ctx = RequestContext(user=_root_ctx().user, role=Role.USER)
    access = AsyncMock()
    if not allowed:
        access.side_effect = PermissionDeniedError("management denied")
    execute = AsyncMock(return_value={"status": "completed"})
    fake = SimpleNamespace(
        viking_fs=SimpleNamespace(_ensure_access=access),
        reindex=execute,
    )
    monkeypatch.setattr("openviking.server.routers.content.get_service", lambda: fake)
    request = ReindexRequest(uri="viking://resources/managed/note.md")
    if allowed:
        result = await reindex(body=request, ctx=ctx)
        assert result.status == "ok"
        assert [call.kwargs["action"] for call in access.await_args_list] == [
            AclAction.READ,
            AclAction.MANAGE,
        ]
        execute.assert_awaited_once()
    else:
        with pytest.raises(PermissionDeniedError):
            await reindex(body=request, ctx=ctx)
        execute.assert_not_awaited()


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "full"},
        {"tags": ["x=y"]},
        {"tag_mode": "clear"},
    ],
)
async def test_resource_exception_does_not_allow_other_processing(monkeypatch, options):
    ctx = RequestContext(user=_root_ctx().user, role=Role.USER)
    execute = AsyncMock()
    monkeypatch.setattr(
        "openviking.server.routers.content.get_service", lambda: SimpleNamespace(reindex=execute)
    )
    with pytest.raises(PermissionDeniedError):
        await reindex(body=ReindexRequest(uri="viking://resources/managed", **options), ctx=ctx)
    execute.assert_not_awaited()


async def test_real_user_resource_vectors_route(
    admitted_service, app, client, monkeypatch, persisted
):
    service = admitted_service
    root, uri = await _seed(service)
    ctx = RequestContext(user=_root_ctx().user, role=Role.USER)
    app.dependency_overrides[get_request_context] = lambda: ctx
    _forbid_models(monkeypatch)
    response = await asyncio.wait_for(
        client.post(
            "/api/v1/content/reindex",
            json={"uri": root, "mode": "vectors_only", "wait": True, "force": True},
        ),
        timeout=30,
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["status"] == "completed" and result["failed_records"] == 0, result
    assert result["rebuilt_records"] >= 1, result
    jobs = [data for name, data in persisted if "eindex" in name]
    embeddings = [data for name, data in persisted if "mbedding" in name]
    assert jobs and embeddings
    assert all(data["role"] == "user" for data in jobs)
    assert any(uri == data["payload"]["context_data"]["uri"] for data in embeddings)


async def test_real_manage_denial_prevents_task(admitted_service, app, client, persisted):
    service = admitted_service
    root, uri = await _seed(service, indexed=True)
    ctx = RequestContext(user=_root_ctx().user, role=Role.USER)
    await service.runtime_config_manager.patch_account(ctx.account_id, {"acl": {"enabled": True}})
    await service.viking_fs.set_acl(
        root,
        [{"principal": "user:" + ctx.user.user_id, "level": "read"}],
        acl_mode=AclMode.RESTRICTED,
        ctx=_root_ctx(),
    )
    persisted.clear()
    app.dependency_overrides[get_request_context] = lambda: ctx
    response = await client.post("/api/v1/content/reindex", json={"uri": root})
    assert response.status_code == 403, response.text
    assert not persisted


async def test_child_read_without_manage_refuses_whole_plan(
    admitted_service,
    app,
    client,
    monkeypatch,
    persisted,
):
    service = admitted_service
    root, uri = await _seed(service, indexed=True)
    ctx = RequestContext(user=_root_ctx().user, role=Role.USER)
    await service.runtime_config_manager.patch_account(ctx.account_id, {"acl": {"enabled": True}})
    await service.viking_fs.set_acl(
        root,
        [{"principal": "user:" + ctx.user.user_id, "level": "manage"}],
        acl_mode=AclMode.RESTRICTED,
        ctx=_root_ctx(),
    )
    await service.viking_fs.set_acl(
        uri,
        [{"principal": "user:" + ctx.user.user_id, "level": "read"}],
        acl_mode=AclMode.RESTRICTED,
        ctx=_root_ctx(),
    )
    persisted.clear()
    captures = []
    import openviking.storage.queuefs.resource_embedding_fence as fence

    original = fence.capture_resource_index_conditions

    async def capture(*args, **kwargs):
        captures.append(True)
        return await original(*args, **kwargs)

    monkeypatch.setattr(fence, "capture_resource_index_conditions", capture)
    app.dependency_overrides[get_request_context] = lambda: ctx
    try:
        response = await asyncio.wait_for(
            client.post(
                "/api/v1/content/reindex",
                json={"uri": root, "force": True},
            ),
            timeout=30,
        )
        if response.status_code == 200:
            assert response.json()["result"]["failed_records"] > 0, response.text
        else:
            assert response.status_code in (403, 500), response.text
    except PermissionDeniedError:
        pass
    assert not captures
    assert not [data for name, data in persisted if "mbedding" in name]
