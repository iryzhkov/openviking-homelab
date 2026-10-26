# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Admission and final-lease account lifetime checks for qualified mutations."""

from openviking.server.account_stores.models import AccountIdentity
from openviking_cli.exceptions import ConflictError


async def capture_account_identity(manager, account_id: str) -> AccountIdentity:
    """Capture the authoritative lifetime at admission, independent of actor role.

    This helper performs no filesystem or index writes. The caller owns lease
    acquisition/release and must retain its final coverage through mutation.
    Unsupported stores and read failures propagate as refusals.
    """
    identity = await manager.get_account_identity(account_id)
    if (
        not isinstance(identity, AccountIdentity)
        or not isinstance(identity.creation_identity, str)
        or not identity.creation_identity
        or type(identity.deleting) is not bool
        or identity.deleting
    ):
        raise ConflictError("Account is absent, deleting or has no qualified identity")
    return identity


async def recheck_account_identity(
    manager, account_id: str, admitted: AccountIdentity
) -> None:
    """Re-read after final lease acquisition, before content/index/ACL writes.

    ROOT uses this exact path too. On refusal the caller releases coverage but
    must not remove acquisition-created directories, whose ownership is unknown.
    """
    current = await capture_account_identity(manager, account_id)
    if current != admitted:
        raise ConflictError("Account lifetime changed after admission")
