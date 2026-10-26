"""Fresh caller -> persisted queue -> native fenced effects with bounded inventories."""
# ruff: noqa: F811
# Imported pytest fixtures are injected as parameters, shadowing their re-exports.

from dataclasses import replace

import pytest

from openviking.storage.resource_rfv import (
    RESOURCE_REINDEX_ACTION_LIMIT,
    RESOURCE_REINDEX_INVENTORY_LIMIT,
    validate_resource_reindex_uri,
)
from tests.server.conftest import service as service
from tests.server.conftest import temp_dir as temp_dir
from tests.server.test_vectors_only_reindex_admission_e2e import (
    _forbid_models,
    _root_ctx,
    _run,
    _seed,
)
from tests.server.test_vectors_only_reindex_admission_e2e import (
    admitted_service as admitted_service,
)
from tests.server.test_vectors_only_reindex_admission_e2e import (
    persisted as persisted,
)


async def _orphans(service, root, count, *, level=2):
    ctx = _root_ctx()
    backend = service.viking_fs._get_vector_store()
    records = [
        {
            "id": f"bounded-orphan-{i}",
            "uri": f"{root}/orphan-{i}.md",
            "account_id": ctx.account_id,
            "level": level,
            "context_type": "resource",
            "md5": "orphan",
            "abstract": "orphan",
            "vector": [0.1] * 2048,
        }
        for i in range(count)
    ]
    assert len(await backend._upsert_many_raw(records, ctx=ctx)) == count
    return backend


def _effect_recorder(monkeypatch):
    from openviking.utils.resource_processor import ResourceProcessor

    effects = []
    original = ResourceProcessor._enqueue_index_actions

    async def enqueue(self, actions, **kwargs):
        effects.extend(actions)
        return await original(self, actions, **kwargs)

    monkeypatch.setattr(ResourceProcessor, "_enqueue_index_actions", enqueue)
    # Enqueue is the FIRST action boundary, stricter than counting committed writes.
    return effects


@pytest.mark.parametrize(
    "extra",
    [
        "/../outside.md",
        "/%2e%2e/out.md",
        "/{user:alias}",
        "/.path.ovlock",
        "/.exact.ovlock.x",
        "/.relations.json",
        "/x\\y",
        "/space name.md",
        "/a" + "x" * 256,
        "/" + "/".join(["b" * 210] * 5),
        "/bad\u200dname",
    ],
)
async def test_real_queued_inventory_invalid_uri_refuses_before_first_effect(
    admitted_service, monkeypatch, persisted, extra
):
    service = admitted_service
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    root, _ = await _seed(service)
    backend = await _orphans(service, root, 1)
    original = backend._strict_transfer_page

    async def poisoned(*args, **kwargs):
        rows, cursor = await original(*args, **kwargs)
        if rows:
            rows = [dict(row, uri=root + extra) for row in rows]
        return rows, cursor

    monkeypatch.setattr(backend, "_strict_transfer_page", poisoned)
    effects = _effect_recorder(monkeypatch)
    _forbid_models(monkeypatch)
    with pytest.raises(Exception, match="canonical|inventory|invalid|incomplete|limit"):
        await _run(service, root)
    assert not effects
    assert not [data for name, data in persisted if "mbedding" in name]


@pytest.mark.parametrize("location", ["sibling", "ancestor", "broader", "file_descendant"])
async def test_real_queued_inventory_outside_exact_target_refuses(
    admitted_service, monkeypatch, persisted, location
):
    service = admitted_service
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    root, uri = await _seed(service)
    backend = await _orphans(service, root, 1)
    target = uri if location == "file_descendant" else root
    outside = {
        "sibling": root + "-neighbor/a.md",
        "ancestor": "viking://resources",
        "broader": "viking://user/private",
        "file_descendant": uri + "/orphan.md",
    }[location]
    original = backend._strict_transfer_page

    async def poisoned(*args, **kwargs):
        rows, cursor = await original(*args, **kwargs)
        # The exact-file query may have no existing rows. Supply one adversarial
        # response after the real account/path query, keeping the first-effect proof.
        if not rows:
            rows = [{"id": "poison", "level": 2, "md5": "", "account_id": _root_ctx().account_id}]
        return [dict(row, uri=outside) for row in rows], cursor

    monkeypatch.setattr(backend, "_strict_transfer_page", poisoned)
    effects = _effect_recorder(monkeypatch)
    with pytest.raises(Exception, match="canonical|inventory|invalid|incomplete|limit"):
        await _run(service, target)
    assert not effects and not [data for name, data in persisted if "mbedding" in name]


@pytest.mark.parametrize(
    "count,reason",
    [
        (RESOURCE_REINDEX_INVENTORY_LIMIT + 1, "inventory"),
        (RESOURCE_REINDEX_INVENTORY_LIMIT, "action"),
    ],
)
async def test_real_orphan_budget_refuses_complete_operation_before_effect(
    admitted_service, monkeypatch, persisted, count, reason
):
    service = admitted_service
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    root, _ = await _seed(service)
    backend = await _orphans(service, root, count)
    pages = []
    original = backend._strict_transfer_page

    async def page(*args, **kwargs):
        pages.append(kwargs["limit"])
        return await original(*args, **kwargs)

    monkeypatch.setattr(backend, "_strict_transfer_page", page)
    effects = _effect_recorder(monkeypatch)
    with pytest.raises(Exception, match="limit"):
        await _run(service, root)
    assert not effects and not [data for name, data in persisted if "mbedding" in name]
    if reason == "inventory":
        assert not pages  # Count over ceiling refuses before fetching an orphan page.
        # Default callers remain unbounded and retain strict complete pagination.
        inventory = await backend.get_incremental_inventory_under_uri(root, ctx=_root_ctx())
        assert len(inventory) == count
    else:
        assert pages and all(limit <= 385 for limit in pages)


async def test_real_exact_action_ceiling_with_orphans_commits_through_fenced_queue(
    admitted_service, monkeypatch, persisted
):
    service = admitted_service
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    root, uri = await _seed(service)
    await _orphans(service, root, RESOURCE_REINDEX_ACTION_LIMIT - 1)
    effects = _effect_recorder(monkeypatch)
    _forbid_models(monkeypatch)
    import asyncio

    result = await asyncio.wait_for(
        service.reindex(uri=root, mode="vectors_only", force=True, wait=True, ctx=_root_ctx()),
        timeout=100,
    )
    assert result["failed_records"] == 0, result
    assert result["deleted_records"] == RESOURCE_REINDEX_ACTION_LIMIT - 1
    assert len(effects) == RESOURCE_REINDEX_ACTION_LIMIT
    embeddings = [data for name, data in persisted if "mbedding" in name]
    assert len(embeddings) == RESOURCE_REINDEX_ACTION_LIMIT
    assert all(data["payload"]["context_data"]["_resource_fence"]["owner"] for data in embeddings)
    assert (
        await service.viking_fs.read_file_bytes(uri, ctx=_root_ctx())
        == b"# Original source\n\nPreserved content"
    )


@pytest.mark.parametrize("level", [0, 1, 2])
async def test_real_context_only_orphan_levels_are_scoped_and_fenced(
    admitted_service, monkeypatch, persisted, level
):
    service = admitted_service
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    root, _ = await _seed(service)
    await _orphans(service, root, 1, level=level)
    effects = _effect_recorder(monkeypatch)
    result = await _run(service, root)
    assert result["failed_records"] == 0 and result["deleted_records"] == 1
    assert effects
    assert any(data.get("action") == "delete" for name, data in persisted if "mbedding" in name)


@pytest.mark.parametrize("fault", ["truncate", "continuation", "wrong_account"])
async def test_real_bounded_scan_cannot_claim_incomplete_inventory_complete(
    admitted_service, monkeypatch, persisted, fault
):
    service = admitted_service
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    root, _ = await _seed(service)
    backend = await _orphans(service, root, 1)
    original = backend._strict_transfer_page

    async def poisoned(*args, **kwargs):
        rows, cursor = await original(*args, **kwargs)
        if fault == "truncate":
            return [], None
        if fault == "continuation":
            return rows, "unexpected-more"
        return [dict(row, account_id="foreign-account") for row in rows], cursor

    monkeypatch.setattr(backend, "_strict_transfer_page", poisoned)
    effects = _effect_recorder(monkeypatch)
    with pytest.raises(Exception, match="canonical|inventory|invalid|incomplete|limit"):
        await _run(service, root)
    assert not effects and not [data for name, data in persisted if "mbedding" in name]


@pytest.mark.parametrize("name", ["café.md", "cafe\u0301.md", "猫😀.md", "a1²2.md"])
async def test_real_canonical_utf8_name_roundtrip(admitted_service, monkeypatch, name):
    service = admitted_service
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    root, _ = await _seed(service)
    uri = root + "/" + name
    await service.viking_fs.write_file(uri, "unicode content", ctx=_root_ctx())
    _forbid_models(monkeypatch)
    result = await _run(service, uri)
    assert result["failed_records"] == 0, result
    validate_resource_reindex_uri(service.viking_fs, uri, _root_ctx(), target=uri, is_dir=False)


async def test_real_formal_limit_refuses_without_any_index_effect(
    admitted_service, monkeypatch, persisted
):
    service = admitted_service
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    root, _ = await _seed(service)
    for i in range(128):
        await service.viking_fs.write_file(f"{root}/formal-{i}.txt", "body", ctx=_root_ctx())
    effects = _effect_recorder(monkeypatch)
    with pytest.raises(Exception, match="limit"):
        await _run(service, root)
    assert not effects and not [data for name, data in persisted if "mbedding" in name]


async def test_entire_plan_validated_before_capture_even_after_good_first_action(
    admitted_service, monkeypatch, persisted
):
    import openviking.storage.context_update_plan as planning

    service = admitted_service
    monkeypatch.setattr("openviking.server.dependencies._service", service)
    root, _ = await _seed(service)
    original = planning.build_rfv_context_update_plan

    def poisoned(**kwargs):
        diff, plan = original(**kwargs)
        good = plan.direct_index_actions[0]
        bad = replace(good, uri=root + "-foreign/item.md")
        return diff, replace(plan, direct_index_actions=(*plan.direct_index_actions, bad))

    monkeypatch.setattr(planning, "build_rfv_context_update_plan", poisoned)
    effects = _effect_recorder(monkeypatch)
    with pytest.raises(Exception, match="canonical"):
        await _run(service, root)
    assert not effects and not [data for name, data in persisted if "mbedding" in name]
