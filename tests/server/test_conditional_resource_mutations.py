# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Create-only batch writes (expected_absent) and no-replace moves (no_replace).

These run against the in-process service with real RAGFS path locks, so the
conditions are checked under the same locks other backend writers use. Writers
that "win" between the unlocked check and lock acquisition are injected at the
lock call itself.
"""

import asyncio
import hashlib

import pytest
import pytest_asyncio

from openviking.server.api_keys.legacy import FileStore

from openviking.server.identity import RequestContext, Role
from openviking_cli.exceptions import ConflictError
from openviking_cli.session.user_id import UserIdentifier


@pytest_asyncio.fixture(autouse=True)
async def authoritative_account_manager(service):
    """ASGI fixtures bypass application lifespan, so install its real DI explicitly."""
    manager = FileStore(service.viking_fs, watch_enabled=False)
    await manager.load()
    service.viking_fs.set_account_identity_manager(manager)
    try:
        yield manager
    finally:
        await manager.close()


def _root_ctx() -> RequestContext:
    return RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)


async def _seed(client, root: str, files: dict[str, str]) -> None:
    response = await client.post(
        "/api/v1/content/batch-write",
        json={
            "root_uri": root,
            "defer_processing": True,
            "operations": [
                {"uri": uri, "content": body, "mode": "upsert"} for uri, body in files.items()
            ],
        },
    )
    assert response.status_code == 200, response.text


async def _read(client, uri: str) -> str | None:
    response = await client.get("/api/v1/content/read", params={"uri": uri, "raw": True})
    if response.status_code == 404:
        return None
    assert response.status_code == 200, response.text
    return response.json()["result"]


async def _tree(client, uri: str) -> list[str] | None:
    response = await client.get(
        "/api/v1/fs/ls",
        params={"uri": uri, "simple": True, "recursive": True, "output": "original"},
    )
    if response.status_code == 404:
        return None
    assert response.status_code == 200, response.text
    prefix = uri.rstrip("/") + "/"
    return sorted(item.removeprefix(prefix) for item in response.json()["result"])


async def _batch(client, root: str, operations: list[dict], **options):
    return await client.post(
        "/api/v1/content/batch-write",
        json={"root_uri": root, "defer_processing": True, "operations": operations, **options},
    )


# -- batch-write expected_absent ---------------------------------------------


async def test_expected_absent_creates_missing_file(client_with_resource):
    client, root = client_with_resource
    target = f"{root}/create-only.md"
    response = await _batch(
        client, root, [{"uri": target, "content": "first", "mode": "create", "expected_absent": True}]
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"]["created"] == [target]
    assert await _read(client, target) == "first"


async def test_expected_absent_existing_file_conflicts_before_any_write(client_with_resource):
    client, root = client_with_resource
    existing = f"{root}/occupied.md"
    fresh = f"{root}/a-would-be-created.md"
    await _seed(client, root, {existing: "keep me"})

    response = await _batch(
        client,
        root,
        [
            # Sorted first, so a check-as-you-go implementation would write it.
            {"uri": fresh, "content": "new", "mode": "create", "expected_absent": True},
            {"uri": existing, "content": "clobber", "mode": "create", "expected_absent": True},
        ],
    )
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "CONFLICT"
    assert await _read(client, existing) == "keep me"
    assert await _read(client, fresh) is None


async def test_expected_absent_existing_directory_conflicts(client_with_resource):
    client, root = client_with_resource
    directory = f"{root}/folder"
    await _seed(client, root, {f"{directory}/child.md": "child"})

    response = await _batch(
        client,
        root,
        [{"uri": directory, "content": "x", "mode": "create", "expected_absent": True}],
    )
    assert response.status_code == 409, response.text
    assert await _tree(client, directory) == ["child.md"]
    assert await _read(client, f"{directory}/child.md") == "child"


async def test_expected_absent_skip_conflicts_reports_and_keeps_existing(client_with_resource):
    client, root = client_with_resource
    existing = f"{root}/occupied.md"
    fresh = f"{root}/fresh.md"
    await _seed(client, root, {existing: "keep me"})

    response = await _batch(
        client,
        root,
        [
            {"uri": existing, "content": "clobber", "expected_absent": True},
            {"uri": fresh, "content": "new", "expected_absent": True},
        ],
        skip_conflicts=True,
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["created"] == [fresh]
    assert [item["uri"] for item in result["conflicts"]] == [existing]
    assert await _read(client, existing) == "keep me"


async def test_expected_absent_and_expected_sha256_are_mutually_exclusive(client_with_resource):
    client, root = client_with_resource
    target = f"{root}/both.md"
    response = await _batch(
        client,
        root,
        [
            {
                "uri": target,
                "content": "x",
                "expected_absent": True,
                "expected_sha256": hashlib.sha256(b"").hexdigest(),
            }
        ],
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "INVALID_ARGUMENT"
    assert await _read(client, target) is None


async def test_create_without_expected_absent_keeps_overwrite_behavior(client_with_resource):
    client, root = client_with_resource
    existing = f"{root}/legacy-create.md"
    await _seed(client, root, {existing: "old"})
    response = await _batch(client, root, [{"uri": existing, "content": "new", "mode": "create"}])
    assert response.status_code == 200, response.text
    assert response.json()["result"]["updated"] == [existing]
    assert await _read(client, existing) == "new"


async def test_expected_absent_sees_writer_that_won_before_the_lock(
    client_with_resource, service, monkeypatch
):
    """A writer that lands just before the batch lease is detected under the lease."""
    client, root = client_with_resource
    target = f"{root}/raced.md"
    agfs = service.viking_fs._async_agfs
    original = agfs.pathlock_acquire_batch
    raced = False

    async def acquire_after_competing_write(paths, *args, **kwargs):
        nonlocal raced
        if not raced:
            raced = True
            # Ordinary backend write with its own automatic path lock.
            await service.viking_fs.write_file(target, "competitor", ctx=_root_ctx())
        return await original(paths, *args, **kwargs)

    monkeypatch.setattr(agfs, "pathlock_acquire_batch", acquire_after_competing_write)
    response = await _batch(
        client, root, [{"uri": target, "content": "mine", "mode": "create", "expected_absent": True}]
    )
    assert raced
    assert response.status_code == 409, response.text
    assert await _read(client, target) == "competitor"


async def test_expected_absent_concurrent_creators_exactly_one_wins(client_with_resource):
    client, root = client_with_resource
    target = f"{root}/contended.md"
    bodies = [f"writer-{index}" for index in range(4)]
    responses = await asyncio.gather(
        *(
            _batch(
                client,
                root,
                [{"uri": target, "content": body, "mode": "create", "expected_absent": True}],
            )
            for body in bodies
        )
    )
    statuses = [response.status_code for response in responses]
    assert statuses.count(200) == 1, [response.text for response in responses]
    assert statuses.count(409) == len(bodies) - 1
    winner = bodies[statuses.index(200)]
    assert await _read(client, target) == winner


# -- fs/mv no_replace ----------------------------------------------------------


async def _mv(client, source: str, target: str, **options):
    return await client.post(
        "/api/v1/fs/mv", json={"from_uri": source, "to_uri": target, **options}
    )


async def test_mv_no_replace_moves_file_to_absent_destination(client_with_resource):
    client, root = client_with_resource
    source = f"{root}/move-me.md"
    target = f"{root}/moved.md"
    await _seed(client, root, {source: "payload"})
    response = await _mv(client, source, target, no_replace=True)
    assert response.status_code == 200, response.text
    assert response.json()["result"]["no_replace"] is True
    assert await _read(client, target) == "payload"
    assert await _read(client, source) is None


async def test_mv_no_replace_refuses_existing_file(client_with_resource):
    client, root = client_with_resource
    source = f"{root}/src.md"
    target = f"{root}/dst.md"
    await _seed(client, root, {source: "source", target: "destination"})
    response = await _mv(client, source, target, no_replace=True)
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "CONFLICT"
    assert await _read(client, source) == "source"
    assert await _read(client, target) == "destination"


async def test_mv_no_replace_refuses_existing_directory_without_merge(client_with_resource):
    client, root = client_with_resource
    source = f"{root}/srcdir"
    target = f"{root}/dstdir"
    await _seed(
        client,
        root,
        {
            f"{source}/a.md": "source a",
            f"{source}/shared.md": "source shared",
            f"{target}/shared.md": "destination shared",
            f"{target}/z.md": "destination z",
        },
    )
    response = await _mv(client, source, target, no_replace=True)
    assert response.status_code == 409, response.text
    assert await _tree(client, source) == ["a.md", "shared.md"]
    assert await _tree(client, target) == ["shared.md", "z.md"]
    assert await _read(client, f"{target}/shared.md") == "destination shared"


async def test_mv_no_replace_refuses_existing_empty_directory(client_with_resource):
    client, root = client_with_resource
    source = f"{root}/srcdir"
    target = f"{root}/emptydir"
    await _seed(client, root, {f"{source}/a.md": "a"})
    created = await client.post("/api/v1/fs/mkdir", json={"uri": target})
    assert created.status_code == 200, created.text
    response = await _mv(client, source, target, no_replace=True)
    assert response.status_code == 409, response.text
    assert await _tree(client, source) == ["a.md"]


async def test_mv_no_replace_moves_multi_chunk_directory_to_absent_destination(
    client_with_resource,
):
    """Root coverage leaves the destination absent until exclusive creation."""
    client, root = client_with_resource
    source = f"{root}/chunked"
    target = f"{root}/chunked-moved"
    files = {f"{source}/part-{index:02d}.md": f"chunk {index}" for index in range(6)}
    files[f"{source}/nested/inner.md"] = "inner"
    await _seed(client, root, files)
    response = await _mv(client, source, target, no_replace=True)
    assert response.status_code == 200, response.text
    moved = await _tree(client, target)
    assert [name for name in moved if name.endswith(".md")] == sorted(
        uri.removeprefix(f"{source}/") for uri in files
    )
    assert await _read(client, f"{target}/nested/inner.md") == "inner"
    assert await _tree(client, source) is None


async def test_mv_default_still_overwrites_existing_file(client_with_resource):
    client, root = client_with_resource
    source = f"{root}/legacy-src.md"
    target = f"{root}/legacy-dst.md"
    await _seed(client, root, {source: "source", target: "destination"})
    response = await _mv(client, source, target)
    assert response.status_code == 200, response.text
    assert "no_replace" not in response.json()["result"]
    assert await _read(client, target) == "source"


@pytest.mark.parametrize("is_dir", [False, True])
async def test_mv_no_replace_sees_writer_that_won_before_the_lock(
    client_with_resource, service, monkeypatch, is_dir
):
    """The recheck under the move's own locks catches a destination created after
    the unlocked check, and neither side loses data."""
    client, root = client_with_resource
    if is_dir:
        source, target = f"{root}/race-src", f"{root}/race-dst"
        await _seed(client, root, {f"{source}/a.md": "source a"})
        competitor = f"{target}/competitor.md"
    else:
        source, target = f"{root}/race-src.md", f"{root}/race-dst.md"
        await _seed(client, root, {source: "source"})
        competitor = target
    viking_fs = service.viking_fs
    agfs = viking_fs._async_agfs
    original = agfs.pathlock_acquire_batch
    raced = False

    async def acquire_after_competing_write(requests, *args, **kwargs):
        nonlocal raced
        if not raced:
            raced = True
            await viking_fs.write_file(competitor, "competitor", ctx=_root_ctx())
        return await original(requests, *args, **kwargs)

    monkeypatch.setattr(agfs, "pathlock_acquire_batch", acquire_after_competing_write)
    # Call the storage move directly so its locked recheck is the only one left.
    with pytest.raises(ConflictError):
        await viking_fs.mv(source, target, ctx=_root_ctx(), no_replace=True)
    monkeypatch.setattr(agfs, "pathlock_acquire_batch", original)
    assert raced
    assert await _read(client, competitor) == "competitor"
    if is_dir:
        assert await _tree(client, target) == ["competitor.md"]
        assert await _read(client, f"{source}/a.md") == "source a"
    else:
        assert await _read(client, source) == "source"


async def test_mv_no_replace_concurrent_moves_exactly_one_wins(client_with_resource):
    client, root = client_with_resource
    target = f"{root}/single-slot.md"
    sources = {f"{root}/candidate-{index}.md": f"candidate {index}" for index in range(3)}
    await _seed(client, root, sources)
    responses = await asyncio.gather(
        *(_mv(client, source, target, no_replace=True) for source in sources)
    )
    statuses = [response.status_code for response in responses]
    assert statuses.count(200) == 1, [response.text for response in responses]
    assert statuses.count(409) == len(sources) - 1
    winner = list(sources)[statuses.index(200)]
    assert await _read(client, target) == sources[winner]
    for source, body in sources.items():
        if source != winner:
            assert await _read(client, source) == body


async def test_mv_no_replace_and_create_only_write_race_has_one_winner(client_with_resource):
    client, root = client_with_resource
    source = f"{root}/mover.md"
    target = f"{root}/slot.md"
    await _seed(client, root, {source: "moved body"})
    move, write = await asyncio.gather(
        _mv(client, source, target, no_replace=True),
        _batch(
            client,
            root,
            [{"uri": target, "content": "written body", "expected_absent": True}],
        ),
    )
    assert sorted([move.status_code, write.status_code]) == [200, 409], (move.text, write.text)
    if move.status_code == 200:
        assert await _read(client, target) == "moved body"
        assert await _read(client, source) is None
    else:
        assert await _read(client, target) == "written body"
        assert await _read(client, source) == "moved body"


@pytest.mark.parametrize("source_kind", ["file", "directory", "missing"])
async def test_mv_rechecks_source_type_under_final_lease(
    client_with_resource, service, monkeypatch, source_kind
):
    client, root = client_with_resource
    source, target = f"{root}/type-race-source", f"{root}/type-race-target"
    original_is_dir = source_kind == "directory"
    await _seed(client, root, {
        f"{source}/child.md" if original_is_dir else source: "original",
    })
    viking_fs = service.viking_fs
    agfs = viking_fs._async_agfs
    source_path = viking_fs._uri_to_path(source, ctx=_root_ctx())
    original = agfs.pathlock_acquire_batch
    raced = False

    async def acquire_after_source_change(requests, *args, **kwargs):
        nonlocal raced
        if not raced:
            raced = True
            await agfs.rm(source_path, recursive=original_is_dir)
            if source_kind == "file":
                await agfs.mkdir(source_path)
            elif source_kind == "directory":
                await agfs.write(source_path, b"competing file")
        return await original(requests, *args, **kwargs)

    monkeypatch.setattr(agfs, "pathlock_acquire_batch", acquire_after_source_change)
    with pytest.raises(ConflictError):
        await viking_fs.mv(source, target, ctx=_root_ctx(), vector_mode="none", no_replace=True)
    assert raced
    if source_kind == "directory":
        assert await _read(client, source) == "competing file"
    elif source_kind == "file":
        assert await _tree(client, source) == []
    else:
        assert await _read(client, source) is None
    assert await _read(client, target) is None


# -- fs/replace-tree -----------------------------------------------------------


async def _revision(client, uri: str) -> str:
    response = await client.get("/api/v1/fs/tree-revision", params={"uri": uri})
    assert response.status_code == 200, response.text
    return response.json()["result"]["revision"]


async def _snapshot(client, uri: str) -> dict[str, str] | None:
    names = await _tree(client, uri)
    if names is None:
        return None
    files = {}
    for name in names:
        response = await client.get(
            "/api/v1/content/read", params={"uri": f"{uri}/{name}", "raw": True}
        )
        if response.status_code == 400 and response.json()["error"]["details"].get(
            "actual"
        ) == "directory":
            files[name] = "<dir>"
            continue
        assert response.status_code == 200, response.text
        files[name] = response.json()["result"]
    return files


async def _replace(client, **body):
    return await client.post("/api/v1/fs/replace-tree", json=body)


async def _replacement_fixture(client, root: str):
    target, staging, backup = f"{root}/doc", f"{root}/doc-staging", f"{root}/doc-backup"
    await _seed(
        client,
        root,
        {
            f"{target}/chunk-01.md": "old one",
            f"{target}/chunk-02.md": "old two",
            f"{target}/assets/note.md": "old nested",
            f"{staging}/chunk-01.md": "new one",
            f"{staging}/chunk-02.md": "new two",
            f"{staging}/chunk-03.md": "new three",
        },
    )
    return target, staging, backup


async def test_replace_tree_installs_staging_and_keeps_verified_backup(client_with_resource):
    client, root = client_with_resource
    target, staging, backup = await _replacement_fixture(client, root)
    old_snapshot = await _snapshot(client, target)
    new_snapshot = await _snapshot(client, staging)
    old_revision = await _revision(client, target)
    new_revision = await _revision(client, staging)

    response = await _replace(
        client,
        target_uri=target,
        staging_uri=staging,
        backup_uri=backup,
        expected_target_revision=old_revision,
        expected_staging_revision=new_revision,
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["content_committed"] is True
    assert result["previous_revision"] == old_revision
    assert result["revision"] == new_revision
    assert result["cleanup"] == {"status": "completed", "retained": [backup], "errors": []}
    assert result["backup_retained"] is True
    assert await _snapshot(client, target) == new_snapshot
    assert await _revision(client, target) == new_revision
    assert await _snapshot(client, backup) == old_snapshot
    assert await _revision(client, backup) == old_revision
    assert await _tree(client, staging) is None


async def test_replace_tree_discards_backup_only_when_asked(client_with_resource):
    client, root = client_with_resource
    target, staging, backup = await _replacement_fixture(client, root)
    response = await _replace(
        client,
        target_uri=target,
        staging_uri=staging,
        backup_uri=backup,
        expected_target_revision=await _revision(client, target),
        expected_staging_revision=await _revision(client, staging),
        retain_backup=False,
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"]["backup_retained"] is False
    assert await _tree(client, backup) is None


async def test_replace_tree_revision_covers_names_types_and_bytes(client_with_resource, service):
    client, root = client_with_resource
    tree = f"{root}/identity"
    await _seed(client, root, {f"{tree}/a.md": "alpha", f"{tree}/sub/b.md": "beta"})
    seen = {await _revision(client, tree)}

    await _seed(client, root, {f"{tree}/a.md": "alphA"})  # same size, other bytes
    seen.add(await _revision(client, tree))
    moved = await _mv(client, f"{tree}/a.md", f"{tree}/renamed.md")  # same bytes, new name
    assert moved.status_code == 200, moved.text
    seen.add(await _revision(client, tree))
    await _seed(client, root, {f"{tree}/a.md/inner.md": "alphA"})  # old name is now a dir
    seen.add(await _revision(client, tree))
    # Semantic sidecars are written by the backend's own processors, not by clients.
    await service.viking_fs.write_file(f"{tree}/sub/.abstract.md", "sidecar", ctx=_root_ctx())
    seen.add(await _revision(client, tree))
    assert len(seen) == 5
    assert (await client.get("/api/v1/fs/tree-revision", params={"uri": f"{root}/nowhere"})).json()[
        "result"
    ]["revision"] == "absent"


@pytest.mark.parametrize("stale", ["target", "staging", "backup"])
async def test_replace_tree_stale_precondition_conflicts_before_any_change(
    client_with_resource, stale
):
    client, root = client_with_resource
    target, staging, backup = await _replacement_fixture(client, root)
    old_revision = await _revision(client, target)
    new_revision = await _revision(client, staging)
    if stale == "target":
        await _seed(client, root, {f"{target}/assets/note.md": "concurrent edit"})
    elif stale == "staging":
        await _seed(client, root, {f"{staging}/chunk-04.md": "late chunk"})
    else:
        await _seed(client, root, {f"{backup}/someone-else.md": "occupied"})
    before = {uri: await _snapshot(client, uri) for uri in (target, staging, backup)}

    response = await _replace(
        client,
        target_uri=target,
        staging_uri=staging,
        backup_uri=backup,
        expected_target_revision=old_revision,
        expected_staging_revision=new_revision,
    )
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "CONFLICT"
    assert {uri: await _snapshot(client, uri) for uri in (target, staging, backup)} == before


async def test_replace_tree_absent_target_is_exclusively_created_without_backup_root(
    client_with_resource,
):
    client, root = client_with_resource
    target, staging, backup = f"{root}/fresh", f"{root}/fresh-staging", f"{root}/fresh-backup"
    await _seed(client, root, {f"{staging}/a.md": "a"})
    response = await _replace(
        client,
        target_uri=target,
        staging_uri=staging,
        backup_uri=backup,
        expected_target_revision="absent",
        expected_staging_revision=await _revision(client, staging),
    )
    assert response.status_code == 200, response.text
    assert await _snapshot(client, target) == {"a.md": "a"}
    assert await _tree(client, backup) is None

    # An absent expectation fails once the target exists, and leaves no backup behind.
    await _seed(client, root, {f"{staging}/b.md": "b"})
    response = await _replace(
        client,
        target_uri=target,
        staging_uri=staging,
        backup_uri=backup,
        expected_target_revision="absent",
        expected_staging_revision=await _revision(client, staging),
    )
    assert response.status_code == 409, response.text
    assert await _tree(client, backup) is None


async def test_replace_tree_refuses_resources_root_and_other_namespaces(client_with_resource):
    client, root = client_with_resource
    for target in ("viking://resources", "viking://user/default/memories/x"):
        response = await _replace(
            client,
            target_uri=target,
            staging_uri=f"{root}/s",
            backup_uri=f"{root}/b",
            expected_target_revision="absent",
            expected_staging_revision="absent",
        )
        assert response.status_code in (400, 403), response.text


async def _direct_replace(service, target, staging, backup, fault):
    from openviking.service import tree_replace

    viking_fs = service.viking_fs
    ctx = _root_ctx()
    return await tree_replace.replace_tree(
        viking_fs,
        target_uri=target,
        staging_uri=staging,
        backup_uri=backup,
        expected_target_revision=(await tree_replace.tree_revision(viking_fs, target, ctx=ctx))[
            "revision"
        ],
        expected_staging_revision=(await tree_replace.tree_revision(viking_fs, staging, ctx=ctx))[
            "revision"
        ],
        ctx=ctx,
        fault=fault,
    )


@pytest.mark.parametrize("phase", ["backup", "clear_target", "install"])
async def test_replace_tree_failure_restores_old_tree_and_retains_copies(
    client_with_resource, service, phase
):
    from openviking_cli.exceptions import OpenVikingError

    client, root = client_with_resource
    target, staging, backup = await _replacement_fixture(client, root)
    old_snapshot = await _snapshot(client, target)
    old_revision = await _revision(client, target)
    new_snapshot = await _snapshot(client, staging)
    new_revision = await _revision(client, staging)
    if phase == "install":
        # Fail after the target was cleared and part of staging installed.
        from openviking.server import skill_package_update
        from openviking.service import tree_replace

        original = skill_package_update.transfer_skill_package

        async def partial_install(viking_fs, source_uri, target_uri, **kwargs):
            if source_uri == staging:
                await viking_fs.write_file(f"{target}/chunk-01.md", "new one", ctx=_root_ctx(),
                                           lease_ref=kwargs["lease_ref"])
                raise OSError("injected install failure")
            return await original(viking_fs, source_uri, target_uri, **kwargs)

        tree_replace.transfer_skill_package, saved = partial_install, tree_replace.transfer_skill_package
        fault = None
    else:
        saved = None

        async def fault(name):
            if name == phase:
                raise OSError(f"injected {phase} failure")

    try:
        with pytest.raises(OpenVikingError) as raised:
            await _direct_replace(service, target, staging, backup, fault)
    finally:
        if saved is not None:
            from openviking.service import tree_replace

            tree_replace.transfer_skill_package = saved
    details = raised.value.details
    assert details["content_committed"] is False
    # A failure before the target is cleared leaves it untouched; later ones roll back.
    assert details["state"] == ("rolled_back" if phase == "install" else "target_unchanged")
    assert details["target_revision"] == old_revision
    assert await _snapshot(client, target) == old_snapshot
    assert await _snapshot(client, staging) == new_snapshot
    assert await _revision(client, staging) == new_revision
    retained = {item["uri"]: item["revision"] for item in details["retained"]}
    assert retained[staging] == new_revision
    if phase == "backup":
        # An exclusively owned empty backup is retained, never inferred disposable.
        assert await _tree(client, backup) == []
        assert retained[backup].startswith("tree-sha256:")
    else:
        assert retained[backup] == old_revision


async def test_replace_tree_rollback_failure_reports_verified_copies(
    client_with_resource, service, monkeypatch
):
    from openviking.service import tree_replace
    from openviking_cli.exceptions import OpenVikingError

    client, root = client_with_resource
    target, staging, backup = await _replacement_fixture(client, root)
    old_snapshot = await _snapshot(client, target)
    old_revision = await _revision(client, target)
    new_revision = await _revision(client, staging)
    original = tree_replace.transfer_skill_package

    async def failing_transfer(viking_fs, source_uri, target_uri, **kwargs):
        if source_uri in (staging, backup):
            raise OSError("injected transfer failure")
        return await original(viking_fs, source_uri, target_uri, **kwargs)

    monkeypatch.setattr(tree_replace, "transfer_skill_package", failing_transfer)
    with pytest.raises(OpenVikingError) as raised:
        await _direct_replace(service, target, staging, backup, None)
    details = raised.value.details
    assert raised.value.code == "DATA_LOSS"
    assert details["state"] == "rollback_failed"
    assert details["content_committed"] is False
    retained = {item["uri"]: item["revision"] for item in details["retained"]}
    assert retained == {backup: old_revision, staging: new_revision}
    assert await _snapshot(client, backup) == old_snapshot


async def test_replace_tree_lease_blocks_other_backend_writers(client_with_resource, service):
    client, root = client_with_resource
    target, staging, backup = await _replacement_fixture(client, root)
    outside = f"{root}/unrelated.md"
    observed = {}

    async def competing_writers(name):
        if name != "install":
            return
        observed["write"] = (
            await _batch(client, root, [{"uri": f"{target}/chunk-01.md", "content": "intruder"}])
        ).status_code
        observed["create"] = (
            await _batch(
                client, root, [{"uri": f"{target}/late.md", "content": "x", "expected_absent": True}]
            )
        ).status_code
        observed["mv_in"] = (await _mv(client, outside, f"{staging}/unrelated.md")).status_code
        observed["rm"] = (
            await client.request(
                "DELETE", "/api/v1/fs", params={"uri": f"{staging}/chunk-01.md"}
            )
        ).status_code
        observed["unrelated"] = (
            await _batch(client, root, [{"uri": f"{root}/elsewhere.md", "content": "ok"}])
        ).status_code

    await _seed(client, root, {outside: "outside"})
    new_snapshot = await _snapshot(client, staging)
    result = await _direct_replace(service, target, staging, backup, competing_writers)
    assert result["content_committed"] is True
    assert observed["write"] == 409
    assert observed["create"] == 409
    assert observed["mv_in"] == 409
    assert observed["rm"] in (409, 423)
    # The accepted first prerequisite serializes the entire resources namespace.
    assert observed["unrelated"] == 409
    assert await _snapshot(client, target) == new_snapshot
    assert await _read(client, outside) == "outside"


async def test_replace_tree_finishes_after_caller_cancellation(
    client_with_resource, service, monkeypatch
):
    from openviking.service import tree_replace

    client, root = client_with_resource
    target, staging, backup = await _replacement_fixture(client, root)
    new_snapshot = await _snapshot(client, staging)
    old_revision = await _revision(client, target)
    new_revision = await _revision(client, staging)
    reached = asyncio.Event()
    release = asyncio.Event()
    original = tree_replace.replace_tree

    async def paused_replace(*args, **kwargs):
        async def fault(name):
            if name == "install":
                reached.set()
                await release.wait()

        return await original(*args, fault=fault, **kwargs)

    monkeypatch.setattr(tree_replace, "replace_tree", paused_replace)
    call = asyncio.create_task(
        service.fs.replace_tree(
            target_uri=target,
            staging_uri=staging,
            backup_uri=backup,
            expected_target_revision=old_revision,
            expected_staging_revision=new_revision,
            ctx=_root_ctx(),
        )
    )
    await asyncio.wait_for(reached.wait(), 30)
    call.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await call
    for _ in range(200):
        if await _tree(client, staging) is None:
            break
        await asyncio.sleep(0.05)
    assert await _snapshot(client, target) == new_snapshot
    assert await _revision(client, backup) == old_revision
