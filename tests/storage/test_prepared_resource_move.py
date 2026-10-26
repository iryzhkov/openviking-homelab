"""Native policy ordering for qualified resource moves."""

import asyncio
from dataclasses import replace

import pytest
import pytest_asyncio

from openviking.server.error_mapping import is_not_found_error
from openviking.server.identity import RequestContext, Role
from openviking.storage.acl import AclAction, AclManager, AclMode
from openviking.storage.expr import PathScope
from openviking.storage.vector_ids import vector_record_id
from openviking_cli.exceptions import (
    ConflictError,
    FailedPreconditionError,
    OpenVikingError,
    PermissionDeniedError,
)
from openviking_cli.session.user_id import UserIdentifier
from tests.storage.test_transfer_merge_binding import (
    _EnabledAclConfig,
    root_ctx,
)
from tests.storage.test_transfer_merge_binding import (
    binding_fs as binding_fs,
)
from tests.storage.test_transfer_merge_binding import (
    indexed_fs as indexed_fs,
)
from tests.storage.test_tree_replace_structured import enabled_tree as enabled_tree


def user(name):
    return RequestContext(user=UserIdentifier(root_ctx().account_id, name), role=Role.USER)


@pytest_asyncio.fixture
async def private_move(indexed_fs):
    fs, backend = indexed_fs
    root = root_ctx()
    fs.acl_manager = backend.acl_manager = AclManager(backend, _EnabledAclConfig())
    await fs.mkdir("viking://resources/public", ctx=root)
    await fs.mkdir("viking://resources/private", ctx=root)
    return fs, backend, root


async def seed(case, directory, *, extra=0, indexed=True):
    fs, backend, root = case
    source = "viking://resources/private/source" + ("" if directory else ".md")
    target = "viking://resources/public/moved" + ("" if directory else ".md")
    content = (
        {"/root.md": b"private root", "/nested/leaf.md": b"nested private"}
        if directory
        else {"": b"direct private"}
    )
    for suffix, data in content.items():
        await fs.write_file_bytes(source + suffix, data, ctx=root)
    uris = ["viking://resources/private", "viking://resources/public"]
    if indexed:
        uris += [source]
        if directory:
            uris += [source + "/nested", *(source + suffix for suffix in content)]
        uris += [source + f"/orphan-{i}.md" for i in range(extra)]
    records = [
        {
            "id": vector_record_id(root.account_id, uri, 2),
            "uri": uri,
            "level": 2,
            "account_id": root.account_id,
            "vector": [0.1] * 4,
            "abstract": "preserved",
            "search_tags": ["team=private"],
            "tags": '{"keep":"value"}',
        }
        for uri in uris
    ]
    assert len(await backend._upsert_many_raw(records, ctx=root)) == len(records)
    grants = [
        {"principal": "user:" + root.user.user_id, "level": "manage"},
        {"principal": "user:manager", "level": "manage"},
        {"principal": "user:reader", "level": "read"},
    ]
    # A directory inherits private policy from its parent; a file has direct policy.
    await fs.set_acl(
        "viking://resources/private" if directory or not indexed else source,
        grants,
        ctx=root,
        acl_mode=AclMode.RESTRICTED,
    )
    if directory and indexed:
        await fs.set_acl(
            source + "/nested",
            [
                {"principal": "user:" + root.user.user_id, "level": "manage"},
                {"principal": "user:manager", "level": "manage"},
                {"principal": "user:nested-reader", "level": "read"},
            ],
            ctx=root,
            acl_mode=AclMode.RESTRICTED,
        )
    return source, target, content


async def invisible(fs, backend, uri):
    reader = user("stranger")
    try:
        await fs.read_file_bytes(uri, ctx=reader)
    except PermissionDeniedError:
        pass
    except Exception as exc:
        assert is_not_found_error(exc), repr(exc)
    else:
        pytest.fail("Denied reader observed moved bytes")
    records = await backend.search_by_random(
        filter=PathScope("uri", uri, depth=-1), limit=1000, output_fields=["id", "uri"], ctx=reader
    )
    assert records == []


@pytest.mark.parametrize("directory", [False, True])
async def test_policy_precedes_bytes_and_source_policy_outlives_bytes(
    private_move, monkeypatch, directory
):
    fs, backend, root = private_move
    source, target, content = await seed(private_move, directory, extra=105 if directory else 0)
    manager = user("manager")
    await fs._ensure_access(source, manager, action=AclAction.MANAGE)
    old_leaf = source + ("/root.md" if directory else "")
    new_leaf = target + ("/root.md" if directory else "")
    with pytest.raises(PermissionDeniedError):
        await fs.read_file_bytes(old_leaf, ctx=user("stranger"))
    original_publish, original_cp = backend._upsert_many_raw, fs._async_agfs.cp
    original_delete = backend._strict_transfer_delete
    events = []

    async def inspect():
        await invisible(fs, backend, old_leaf)
        await invisible(fs, backend, new_leaf)
        if directory:
            await invisible(fs, backend, source + "/nested/leaf.md")
            await invisible(fs, backend, target + "/nested/leaf.md")

    async def publish(records, **kwargs):
        result = await original_publish(records, **kwargs)
        if any(r["uri"].startswith(target) for r in records):
            events.append("policy")
            await inspect()
        return result

    async def copy(old, new, **kwargs):
        result = await original_cp(old, new, **kwargs)
        if str(new).startswith(fs._uri_to_path(target, ctx=root)):
            events.append("copy")
            await inspect()
            # A source-authorized reader can read destination immediately.
            suffix = str(new)[len(fs._uri_to_path(target, ctx=root)) :]
            who = user("nested-reader") if "/nested/" in suffix else user("reader")
            assert await fs.read_file_bytes(target + suffix, ctx=who)
        return result

    async def delete(ctx, ids):
        events.append("source_policy_cleanup")
        assert not await fs.exists(source, ctx=root)
        await invisible(fs, backend, new_leaf)
        return await original_delete(ctx, ids)

    monkeypatch.setattr(backend, "_upsert_many_raw", publish)
    monkeypatch.setattr(fs._async_agfs, "cp", copy)
    monkeypatch.setattr(backend, "_strict_transfer_delete", delete)
    result = await fs.mv(source, target, ctx=manager, no_replace=True)
    assert result["phase"] == "completed"
    assert events.count("policy") == (2 if directory else 1)
    assert events.index("copy") > max(i for i, e in enumerate(events) if e == "policy")
    assert events[-1] == "source_policy_cleanup"
    assert not await fs.exists(source, ctx=root)
    for suffix, data in content.items():
        who = user("nested-reader") if "/nested/" in suffix else user("reader")
        assert await fs.read_file_bytes(target + suffix, ctx=who) == data
    moved, _ = await backend._scan_uri_transfer_scope(
        replace(root, bypass_acl=True), target, recursive=directory, include_full_records=True
    )
    assert all(
        "team=private" in r["search_tags"] and r["tags"] == '{"keep":"value"}' for r in moved
    )
    assert not (
        await backend._scan_uri_transfer_scope(
            root, source, recursive=directory, include_full_records=True
        )
    )[0]


@pytest.mark.parametrize(
    "contender", ["physical", "record", "missing-policy", "missing-directory-policy"]
)
async def test_refuses_contenders_and_missing_policy_before_effect(private_move, contender):
    fs, backend, root = private_move
    directory = contender == "missing-directory-policy"
    source, target, _ = await seed(
        private_move, directory, indexed=not contender.startswith("missing")
    )
    if contender == "physical":
        await fs.write_file_bytes(target, b"foreign", ctx=root)
    elif contender == "record":
        await backend.upsert(
            {
                "id": vector_record_id(root.account_id, target, 2),
                "uri": target,
                "level": 2,
                "account_id": root.account_id,
                "vector": [0.3] * 4,
                "abstract": "foreign",
            },
            ctx=root,
        )
    before = await backend._scan_uri_transfer_scope(
        root, "viking://resources", recursive=True, include_full_records=True
    )
    with pytest.raises((ConflictError, FailedPreconditionError)):
        await fs.mv(source, target, ctx=user("manager"), no_replace=True)
    assert await fs.read_file_bytes(source + ("/root.md" if directory else ""), ctx=root) == (
        b"private root" if directory else b"direct private"
    )
    assert (await fs.exists(target, ctx=root)) is (contender == "physical")
    if contender == "physical":
        assert await fs.read_file_bytes(target, ctx=root) == b"foreign"
    after = await backend._scan_uri_transfer_scope(
        root, "viking://resources", recursive=True, include_full_records=True
    )
    assert sorted(before[0], key=lambda r: r["id"]) == sorted(after[0], key=lambda r: r["id"])


@pytest.mark.parametrize(
    "fault",
    ["policy", "copy", "cancel", "source-removal", "source-removal-uncertain", "policy-cleanup"],
)
async def test_failure_retains_policy_and_a_complete_copy(private_move, monkeypatch, fault):
    fs, backend, root = private_move
    source, target, _ = await seed(private_move, False)
    publish, copy = backend._upsert_many_raw, fs._async_agfs.cp
    remove, delete = fs._async_agfs.rm, backend._strict_transfer_delete

    async def publish_fault(records, **kwargs):
        result = await publish(records, **kwargs)
        if fault == "policy":
            raise OSError("policy batch response lost")
        return result

    async def copy_fault(old, new, **kwargs):
        result = await copy(old, new, **kwargs)
        if fault == "cancel":
            raise asyncio.CancelledError()
        if fault == "copy":
            raise OSError("copy response lost")
        return result

    async def removal_fault(path, **kwargs):
        if fault == "source-removal" and path == fs._uri_to_path(source, ctx=root):
            raise OSError("source removal refused")
        result = await remove(path, **kwargs)
        if fault == "source-removal-uncertain" and path == fs._uri_to_path(source, ctx=root):
            raise OSError("source removal response lost")
        return result

    async def delete_fault(ctx, ids):
        if fault == "policy-cleanup":
            raise OSError("source policy cleanup unavailable")
        return await delete(ctx, ids)

    monkeypatch.setattr(backend, "_upsert_many_raw", publish_fault)
    monkeypatch.setattr(fs._async_agfs, "cp", copy_fault)
    monkeypatch.setattr(fs._async_agfs, "rm", removal_fault)
    monkeypatch.setattr(backend, "_strict_transfer_delete", delete_fault)
    with pytest.raises(OpenVikingError) as raised:
        await fs.mv(source, target, ctx=user("manager"), no_replace=True)
    assert raised.value.code == "MOVE_INCOMPLETE"
    assert raised.value.details["content_committed"] is (
        fault in {"source-removal", "source-removal-uncertain", "policy-cleanup"}
    )
    expected_removal = None if fault.startswith("source-removal") else fault == "policy-cleanup"
    assert raised.value.details["source_removed"] is expected_removal
    await invisible(fs, backend, source)
    await invisible(fs, backend, target)
    if fault in {"policy-cleanup", "source-removal-uncertain"}:
        assert await fs.read_file_bytes(target, ctx=user("reader")) == b"direct private"
    else:
        assert await fs.read_file_bytes(source, ctx=user("reader")) == b"direct private"
    if fault != "policy":
        assert await fs.read_file_bytes(target, ctx=user("reader")) == b"direct private"


@pytest.mark.parametrize("loss", ["identity", "owner"])
async def test_original_admission_loss_after_policy_refuses_before_bytes(
    private_move, monkeypatch, loss
):
    import json
    from uuid import uuid4

    from openviking.server.api_keys.legacy import ACCOUNTS_PATH

    fs, backend, root = private_move
    source, target, _ = await seed(private_move, False)
    original_acquire = fs._async_agfs.pathlock_acquire_batch
    original_publish = backend._upsert_many_raw
    original_release = fs._async_agfs.pathlock_release
    held = []
    foreign = []

    async def acquire(*args, **kwargs):
        lease = await original_acquire(*args, **kwargs)
        held.append(lease)
        return lease

    async def publish(records, **kwargs):
        result = await original_publish(records, **kwargs)
        if loss == "identity":
            data = json.loads(
                await fs._async_agfs.cat(ACCOUNTS_PATH, fs_ctx={"bypass_cache": "true"})
            )
            data["accounts"][root.account_id]["creation_id"] = str(uuid4())
            await fs._async_agfs.write(ACCOUNTS_PATH, json.dumps(data).encode())
        else:
            assert held
            await original_release(held[-1])
            foreign.append(
                await original_acquire(
                    [{"path": fs._uri_to_path("viking://resources", ctx=root), "kind": "tree"}]
                )
            )
        return result

    monkeypatch.setattr(fs._async_agfs, "pathlock_acquire_batch", acquire)
    monkeypatch.setattr(backend, "_upsert_many_raw", publish)
    try:
        with pytest.raises(OpenVikingError) as raised:
            await fs.mv(source, target, ctx=user("manager"), no_replace=True)
        assert raised.value.code == "MOVE_INCOMPLETE"
        assert raised.value.details["phase"] == "copy_content"
        assert not raised.value.details["content_committed"]
        assert await fs.read_file_bytes(source, ctx=user("reader")) == b"direct private"
        assert not await fs.exists(target, ctx=root)
        if foreign:
            assert await fs._async_agfs.pathlock_refresh(foreign[0]) == "refreshed"
    finally:
        for lease in foreign:
            await original_release(lease)


async def test_file_write_permission_preserves_existing_move_authority(private_move):
    fs, _, root = private_move
    source, target, _ = await seed(private_move, False)
    await fs.set_acl(
        source,
        [
            {"principal": "user:" + root.user.user_id, "level": "manage"},
            {"principal": "user:manager", "level": "write"},
            {"principal": "user:reader", "level": "read"},
        ],
        ctx=root,
        acl_mode=AclMode.RESTRICTED,
    )
    with pytest.raises(PermissionDeniedError):
        await fs._ensure_access(source, user("manager"), action=AclAction.MANAGE)
    await fs._ensure_access(source, user("manager"), action=AclAction.WRITE)
    result = await fs.mv(source, target, ctx=user("manager"), no_replace=True)
    assert result["phase"] == "completed"
    assert await fs.read_file_bytes(target, ctx=user("reader")) == b"direct private"


async def test_private_structured_directory_move_rewrites_sidecars(enabled_tree):
    from openviking.storage.abstract_overview import parse_abstract_overview

    fs, backend, root, source, _, target, old, _ = enabled_tree
    await fs.set_acl(
        source,
        [
            {"principal": "user:" + root.user.user_id, "level": "manage"},
            {"principal": "user:reader", "level": "read"},
        ],
        ctx=root,
        acl_mode=AclMode.RESTRICTED,
    )
    result = await fs.mv(source, target, ctx=root, no_replace=True)
    assert result["phase"] == "completed"
    assert not await fs.exists(source, ctx=root)
    for relative, original in old.items():
        actual = await fs.read_file_bytes(target + "/" + relative, ctx=user("reader"))
        if relative.rsplit("/", 1)[-1] in {".abstract.md", ".overview.md"}:
            document = parse_abstract_overview(actual)
            directory = relative.rpartition("/")[0]
            expected_directory = target + ("/" + directory if directory else "")
            assert document.metadata["directory"] == expected_directory + "/"
            assert source not in document.body
            assert target in document.body
        else:
            assert actual == original
    await invisible(fs, backend, target + "/one.md")
