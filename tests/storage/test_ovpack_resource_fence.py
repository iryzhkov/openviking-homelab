"""Whole-import producer coverage with real native local files and vectors."""

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio

from openviking.server.api_keys.legacy import ACCOUNTS_PATH
from openviking.storage.errors import LockAcquisitionError, ResourceBusyError
from openviking.storage.ovpack.operations import export_ovpack, import_ovpack
from openviking.storage.vector_ids import vector_record_id
from openviking_cli.exceptions import (
    ConflictError,
    FailedPreconditionError,
    InvalidArgumentError,
    OpenVikingError,
)
from openviking_cli.utils.config.embedding_config import EmbeddingConfig
from tests.storage.test_tree_replace_structured import (
    binding_fs as binding_fs,
)
from tests.storage.test_tree_replace_structured import (
    enabled_tree as enabled_tree,
)
from tests.storage.test_tree_replace_structured import (
    indexed_fs as indexed_fs,
)
from tests.storage.test_tree_replace_structured import (
    raw,
)


class Resolver:
    def __init__(self, model="fixture"):
        self.model = model

    async def resolve(self, account_id):
        return SimpleNamespace(
            embedding=EmbeddingConfig(
                dense={
                    "provider": "openai",
                    "model": self.model,
                    "dimension": 4,
                    "api_key": "fixture",
                }
            ),
            vectordb=SimpleNamespace(sparse_weight=0),
        )


@pytest_asyncio.fixture
async def pack(enabled_tree, tmp_path):  # noqa: F811
    fs, backend, ctx, target, *_ = enabled_tree
    source = "viking://resources/source/doc"
    await fs.write_file_bytes(source + "/one.md", b"incoming one", ctx=ctx)
    for name, level, text in [
        (".abstract.md", 0, "incoming abstract"),
        (".overview.md", 1, "incoming overview"),
        ("one.md", 2, "incoming one"),
    ]:
        if level != 2:
            await fs.write_file_bytes(source + "/" + name, text.encode(), ctx=ctx)
        uri = source if level != 2 else source + "/one.md"
        await backend.upsert(
            {
                "id": vector_record_id(ctx.account_id, uri, level),
                "uri": uri,
                "level": level,
                "account_id": ctx.account_id,
                "vector": [0.1] * 4,
                "content": text,
                "abstract": text,
            },
            ctx=ctx,
        )
    resolver = Resolver()
    path = tmp_path / "doc.ovpack"
    await export_ovpack(
        fs,
        source,
        str(path),
        ctx,
        vector_store=backend,
        vector_config_resolver=resolver,
        include_vectors=True,
    )

    async def run(**kwargs):
        return await import_ovpack(
            fs,
            str(path),
            "viking://resources/published",
            ctx,
            on_conflict=kwargs.pop("on_conflict", "overwrite"),
            vector_store=kwargs.pop("vector_store", backend),
            vector_config_resolver=kwargs.pop("resolver", resolver),
            **kwargs,
        )

    return fs, backend, ctx, target, run


async def test_foreign_tree_refuses_before_file_or_index_mutation(pack, monkeypatch):
    fs, backend, ctx, target, run = pack
    before = await raw(fs, target + "/one.md", ctx)
    calls = []
    original = backend.upsert

    async def observe(*args, **kwargs):
        calls.append(True)
        return await original(*args, **kwargs)

    monkeypatch.setattr(backend, "upsert", observe)
    lease = await fs._async_agfs.pathlock_acquire_tree(
        fs._uri_to_path("viking://resources", ctx=ctx)
    )
    try:
        with pytest.raises((LockAcquisitionError, ResourceBusyError)):
            await run()
        assert not calls
        assert await raw(fs, target + "/one.md", ctx) == before
    finally:
        await fs._async_agfs.pathlock_release(lease)


@pytest.mark.parametrize("mode", ["recompute", "auto_fallback"])
async def test_unsupported_deferred_mode_has_zero_effects(pack, monkeypatch, mode):
    fs, backend, ctx, target, run = pack
    before = await raw(fs, target + "/one.md", ctx)
    from openviking.service.tree_replace import _context_records

    previous = await _context_records(fs, target, ctx)
    with pytest.raises(FailedPreconditionError, match="deferred recompute"):
        await run(
            **(
                {"vector_mode": "recompute"}
                if mode == "recompute"
                else {"resolver": Resolver("changed")}
            )
        )
    assert await raw(fs, target + "/one.md", ctx) == before
    assert await _context_records(fs, target, ctx) == previous


@pytest.mark.parametrize("change", ["recreate", "deleting"])
async def test_root_account_identity_rechecked_under_final_coverage(pack, monkeypatch, change):
    fs, _, ctx, target, run = pack
    agfs, acquire = fs._async_agfs, fs._async_agfs.pathlock_acquire_batch
    fired = False

    async def inject(*args, **kwargs):
        nonlocal fired
        lease = await acquire(*args, **kwargs)
        if not fired:
            fired = True
            if change == "recreate":
                data = json.loads(await agfs.cat(ACCOUNTS_PATH))
                data["accounts"][ctx.account_id]["creation_id"] = str(uuid4())
                await agfs.write(ACCOUNTS_PATH, json.dumps(data).encode())
            else:
                await fs._account_identity_manager.begin_deletion(
                    ctx.account_id,
                    None,
                    task_id="fixture",
                    owner_account_id=ctx.account_id,
                    owner_user_id=ctx.user.user_id,
                )
        return lease

    monkeypatch.setattr(agfs, "pathlock_acquire_batch", inject)
    with pytest.raises((ConflictError, FailedPreconditionError)):
        await run()
    assert fired
    assert await raw(fs, target + "/one.md", ctx) == b"old one"


async def test_missing_deeper_parent_not_recreated(pack, monkeypatch):
    fs, _, ctx, target, run = pack
    agfs, acquire = fs._async_agfs, fs._async_agfs.pathlock_acquire_batch
    fired = False

    async def inject(*args, **kwargs):
        nonlocal fired
        if not fired:
            fired = True
            await agfs.rm(fs._uri_to_path("viking://resources/published", ctx=ctx), recursive=True)
        return await acquire(*args, **kwargs)

    monkeypatch.setattr(agfs, "pathlock_acquire_batch", inject)
    with pytest.raises(OpenVikingError):
        await run()
    assert fired and not await fs.exists("viking://resources/published", ctx=ctx)


@pytest.mark.parametrize("conflict", ["fail", "skip"])
async def test_empty_root_race_is_occupied_under_final_coverage(pack, monkeypatch, conflict):
    fs, _, ctx, target, run = pack
    await fs.rm(target, recursive=True, ctx=ctx)
    agfs, acquire = fs._async_agfs, fs._async_agfs.pathlock_acquire_batch
    fired = False

    async def inject(*args, **kwargs):
        nonlocal fired
        lease = await acquire(*args, **kwargs)
        if not fired:
            fired = True
            await agfs.mkdir(
                fs._uri_to_path(target, ctx=ctx), fs_ctx=fs._pathlock_fs_ctx(ctx, lease)
            )
        return lease

    monkeypatch.setattr(agfs, "pathlock_acquire_batch", inject)
    if conflict == "fail":
        with pytest.raises(ConflictError):
            await run(on_conflict=conflict)
    else:
        assert await run(on_conflict=conflict) == target
    assert fired and await fs.exists(target, ctx=ctx)
    assert not await fs.exists(target + "/one.md", ctx=ctx)


async def test_file_and_direct_index_phase_exclude_competing_write_delete(pack, monkeypatch):
    fs, backend, ctx, target, run = pack
    original_write, original_upsert = fs.write_file_bytes, backend.upsert
    phases = []

    async def contend(phase):
        phases.append(phase)
        with pytest.raises((LockAcquisitionError, ResourceBusyError)):
            await original_write(target + "/one.md", b"competitor", ctx=ctx)
        with pytest.raises((LockAcquisitionError, ResourceBusyError)):
            await fs.rm(target, recursive=True, ctx=ctx)

    async def write(*args, **kwargs):
        await contend("file")
        return await original_write(*args, **kwargs)

    async def upsert(*args, **kwargs):
        await contend("index")
        return await original_upsert(*args, **kwargs)

    monkeypatch.setattr(fs, "write_file_bytes", write)
    monkeypatch.setattr(backend, "upsert", upsert)
    assert await run() == target
    assert set(phases) == {"file", "index"}
    assert await raw(fs, target + "/one.md", ctx) == b"incoming one"
    assert not await fs.exists(target + "/branch", ctx=ctx)
    assert await backend.get([vector_record_id(ctx.account_id, target + "/one.md", 2)], ctx=ctx)


async def test_metadata_and_direct_acl_are_retained(pack):
    fs, backend, ctx, target, run = pack
    from openviking.storage.acl import AclMode

    await fs.set_acl(
        target,
        [{"principal": f"user:{ctx.user.user_id}", "level": "manage"}],
        ctx=ctx,
        acl_mode=AclMode.RESTRICTED,
    )
    records = await backend.get([vector_record_id(ctx.account_id, target, 0)], ctx=ctx)
    record = records[0]
    record.update(search_tags=["owner=retained"], tags="retained-tag")
    await backend.upsert(record, ctx=ctx)
    before = await fs.get_acl(target, ctx=ctx)
    await run()
    after = await fs.get_acl(target, ctx=ctx)
    assert before == after
    result = (await backend.get([vector_record_id(ctx.account_id, target, 0)], ctx=ctx))[0]
    assert result["search_tags"] == ["owner=retained"]
    assert result["tags"] == "retained-tag"


async def test_descendant_acl_denial_precedes_all_mutations(pack):
    fs, backend, ctx, target, run = pack
    from openviking.storage.acl import AclMode
    from openviking_cli.exceptions import PermissionDeniedError

    await fs.set_acl(
        target + "/branch",
        [{"principal": "user:someone-else", "level": "manage"}],
        ctx=ctx,
        acl_mode=AclMode.RESTRICTED,
    )
    with pytest.raises(PermissionDeniedError):
        await run()
    assert await raw(fs, target + "/one.md", ctx) == b"old one"
    assert await raw(fs, target + "/branch/two.md", ctx) == b"old two"


async def test_snapshot_without_indexed_root_refuses_before_mutation(pack, tmp_path):
    fs, backend, ctx, target, _ = pack
    source = "viking://resources/unindexed/doc"
    await fs.write_file_bytes(source + "/one.md", b"incoming", ctx=ctx)
    await backend.upsert(
        {
            "id": vector_record_id(ctx.account_id, source + "/one.md", 2),
            "uri": source + "/one.md",
            "level": 2,
            "account_id": ctx.account_id,
            "vector": [0.1] * 4,
            "abstract": "incoming",
        },
        ctx=ctx,
    )
    path = tmp_path / "unindexed.ovpack"
    await export_ovpack(
        fs,
        source,
        str(path),
        ctx,
        vector_store=backend,
        vector_config_resolver=Resolver(),
        include_vectors=True,
    )
    with pytest.raises(InvalidArgumentError, match="indexed root"):
        await import_ovpack(
            fs,
            str(path),
            "viking://resources/published",
            ctx,
            on_conflict="overwrite",
            vector_store=backend,
            vector_config_resolver=Resolver(),
        )
    assert await raw(fs, target + "/one.md", ctx) == b"old one"


async def test_ordinary_nonresource_import_preserves_recompute(pack, tmp_path, monkeypatch):
    fs, backend, ctx, _, _ = pack
    from unittest.mock import AsyncMock

    from openviking.storage.ovpack import operations

    source = f"viking://user/{ctx.user.user_id}/memories/original/source"
    parent = f"viking://user/{ctx.user.user_id}/memories/destination"
    await fs.write_file_bytes(source + "/one.md", b"ordinary", ctx=ctx)
    await fs.mkdir(parent, ctx=ctx)
    path = tmp_path / "ordinary.ovpack"
    await export_ovpack(fs, source, str(path), ctx)
    enqueue = AsyncMock()
    monkeypatch.setattr(operations, "_enqueue_direct_vectorization", enqueue)
    result = await import_ovpack(fs, str(path), parent, ctx, vector_mode="recompute")
    assert result == parent + "/source"
    assert await raw(fs, result + "/one.md", ctx) == b"ordinary"
    enqueue.assert_awaited_once()


async def test_owner_loss_does_not_remove_foreign_root(pack, monkeypatch):
    fs, backend, ctx, target, run = pack
    agfs, original = fs._async_agfs, fs._async_agfs.pathlock_refresh
    foreign = None

    async def lose(lease, **kwargs):
        nonlocal foreign
        if foreign is None:
            await agfs.pathlock_release(lease)
            foreign = await agfs.pathlock_acquire_tree(
                fs._uri_to_path("viking://resources", ctx=ctx)
            )
        return await original(lease, **kwargs)

    monkeypatch.setattr(agfs, "pathlock_refresh", lose)
    try:
        with pytest.raises((FailedPreconditionError, LockAcquisitionError, ResourceBusyError)):
            await run()
        assert foreign is not None
        assert await raw(fs, target + "/one.md", ctx) == b"old one"
        with pytest.raises((LockAcquisitionError, ResourceBusyError)):
            await fs.write_file_bytes(target + "/foreign.md", b"foreign", ctx=ctx)
    finally:
        if foreign is not None:
            await agfs.pathlock_release(foreign)


async def test_late_owner_loss_retains_foreign_root_and_stops_index(pack, monkeypatch):
    fs, backend, ctx, target, run = pack
    agfs, write = fs._async_agfs, fs.write_file_bytes
    foreign = None
    index_calls = []

    async def lose_after_file(uri, data, **kwargs):
        nonlocal foreign
        await write(uri, data, **kwargs)
        if uri == target + "/one.md":
            await agfs.pathlock_release(kwargs["lease_ref"])
            foreign = await agfs.pathlock_acquire_tree(
                fs._uri_to_path("viking://resources", ctx=ctx)
            )
            await agfs.write(
                fs._uri_to_path(target + "/foreign.md", ctx=ctx),
                b"foreign",
                fs_ctx=fs._pathlock_fs_ctx(ctx, foreign),
            )

    async def unexpected_upsert(*args, **kwargs):
        index_calls.append(True)
        raise AssertionError("index started after native ownership loss")

    monkeypatch.setattr(fs, "write_file_bytes", lose_after_file)
    monkeypatch.setattr(backend, "upsert", unexpected_upsert)
    try:
        with pytest.raises(FailedPreconditionError, match="lost its native owner"):
            await run()
        assert foreign is not None
        assert index_calls == []
        assert await raw(fs, target + "/one.md", ctx) == b"incoming one"
        assert await raw(fs, target + "/foreign.md", ctx) == b"foreign"
    finally:
        if foreign is not None:
            await agfs.pathlock_release(foreign)


async def test_absent_target_is_exclusively_created_and_indexed(pack):
    fs, backend, ctx, target, run = pack
    await fs.rm(target, recursive=True, ctx=ctx)
    assert await run() == target
    assert await raw(fs, target + "/one.md", ctx) == b"incoming one"
    record = (
        await backend.get(
            [vector_record_id(ctx.account_id, target + "/one.md", 2)],
            ctx=ctx,
        )
    )[0]
    assert record["vector"] == pytest.approx([0.1] * 4)
    assert record["account_id"] == ctx.account_id


async def test_foreign_vector_store_refused_before_mutation(pack):
    fs, backend, ctx, target, run = pack
    from unittest.mock import AsyncMock

    foreign = SimpleNamespace(upsert=AsyncMock(), get_account_backend=AsyncMock())
    with pytest.raises(FailedPreconditionError, match="account context store"):
        await run(vector_store=foreign)
    foreign.upsert.assert_not_called()
    foreign.get_account_backend.assert_not_called()
    assert await raw(fs, target + "/one.md", ctx) == b"old one"


async def test_index_failure_propagates_without_cleanup(pack, monkeypatch):
    fs, backend, ctx, target, run = pack

    async def fail(*args, **kwargs):
        raise OSError("fixture index failure")

    monkeypatch.setattr(backend, "upsert", fail)
    with pytest.raises(OSError, match="fixture index failure"):
        await run()
    assert await raw(fs, target + "/one.md", ctx) == b"incoming one"
    # Failure retained data and released its lease.
    await fs.write_file_bytes(target + "/after.md", b"after", ctx=ctx)


async def test_cancel_settles_index_before_releasing_coverage(pack, monkeypatch):
    fs, backend, ctx, target, run = pack
    original = backend.upsert
    started, finish = asyncio.Event(), asyncio.Event()

    async def pause(*args, **kwargs):
        started.set()
        await finish.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(backend, "upsert", pause)
    task = asyncio.create_task(run())
    await asyncio.wait_for(started.wait(), 10)
    task.cancel()
    await asyncio.sleep(0)
    with pytest.raises((LockAcquisitionError, ResourceBusyError)):
        await fs.write_file_bytes(target + "/foreign.md", b"foreign", ctx=ctx)
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await raw(fs, target + "/one.md", ctx) == b"incoming one"
    await fs.write_file_bytes(target + "/after.md", b"after", ctx=ctx)
