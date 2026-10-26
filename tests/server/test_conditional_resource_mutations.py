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

from openviking.server.identity import RequestContext, Role
from openviking_cli.exceptions import ConflictError
from openviking_cli.session.user_id import UserIdentifier


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
    assert response.status_code == 422, response.text
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
    original = agfs.pathlock_acquire_exact_batch
    raced = False

    async def acquire_after_competing_write(paths, *args, **kwargs):
        nonlocal raced
        if not raced:
            raced = True
            # Ordinary backend write with its own automatic path lock.
            await service.viking_fs.write_file(target, "competitor", ctx=_root_ctx())
        return await original(paths, *args, **kwargs)

    monkeypatch.setattr(agfs, "pathlock_acquire_exact_batch", acquire_after_competing_write)
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
    """The Tree lock may create the missing destination; that is not an occupant."""
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
