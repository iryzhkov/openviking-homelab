"""Qualified service move keeps policy through native move and parent work."""

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.acl import AclManager, AclMode
from openviking_cli.exceptions import PermissionDeniedError
from openviking_cli.session.user_id import UserIdentifier
from tests.server.conftest import service as service
from tests.server.conftest import temp_dir as temp_dir
from tests.server.test_real_resource_producer_e2e import _creation_identity, _root_ctx
from tests.server.test_real_resource_producer_e2e import admitted_service as admitted_service
from tests.storage.test_transfer_merge_binding import _EnabledAclConfig


async def test_qualified_service_move_preserves_private_policy_and_original_admission(
    admitted_service, monkeypatch
):
    service = admitted_service
    fs = service.viking_fs
    root = _root_ctx()
    source = "viking://resources/prepared-move/old.md"
    target = "viking://resources/prepared-move/new.md"
    await service.fs.write(
        source, "# Private move body", ctx=root, wait=True, tags=["team=private"]
    )
    store = fs._get_vector_store()
    fs.acl_manager = store.acl_manager = AclManager(store, _EnabledAclConfig())
    await fs.set_acl(
        source,
        [
            {"principal": "user:" + root.user.user_id, "level": "manage"},
            {"principal": "user:reader", "level": "read"},
        ],
        ctx=root,
        acl_mode=AclMode.RESTRICTED,
    )
    stranger = RequestContext(user=UserIdentifier(root.account_id, "stranger"), role=Role.USER)
    reader = RequestContext(user=UserIdentifier(root.account_id, "reader"), role=Role.USER)
    with pytest.raises(PermissionDeniedError):
        await fs.read_file_bytes(source, ctx=stranger)
    original_identity = await _creation_identity(service)
    captures = []
    native_admissions = []
    capture = fs._capture_resource_identity
    native_move = fs.mv

    async def capture_once(ctx):
        result = await capture(ctx)
        captures.append(result.creation_identity)
        return result

    async def observe(*args, **kwargs):
        native_admissions.append(kwargs["resource_admission"].creation_identity)
        return await native_move(*args, **kwargs)

    monkeypatch.setattr(fs, "_capture_resource_identity", capture_once)
    monkeypatch.setattr(fs, "mv", observe)
    await service.fs.mv(source, target, ctx=root, no_replace=True)
    assert captures == [original_identity]
    assert native_admissions == [original_identity]
    with pytest.raises(PermissionDeniedError):
        await fs.read_file_bytes(target, ctx=stranger)
    assert await fs.read_file_bytes(target, ctx=reader) == b"# Private move body"
    assert not await fs.exists(source, ctx=root)
