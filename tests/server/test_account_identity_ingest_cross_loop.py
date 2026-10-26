# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Bounded account-read contention during the actual persisted ingest pipeline."""

import asyncio

from tests.server.conftest import sample_markdown_file as sample_markdown_file
from tests.server.conftest import service as service
from tests.server.conftest import temp_dir as temp_dir
from tests.server.test_real_resource_producer_e2e import _root_ctx
from tests.server.test_real_resource_producer_e2e import admitted_service as admitted_service
from tests.storage.test_account_identity_cross_loop import QueueLoop, result


async def test_real_ingest_with_parallel_fresh_account_checks(
    admitted_service, sample_markdown_file
):
    service = admitted_service
    store = service.viking_fs._account_identity_manager
    ctx = _root_ctx()
    admitted = await store.get_account_identity(ctx.account_id)
    observed_loops = set()

    async def checks():
        observed_loops.add(asyncio.get_running_loop())
        for _ in range(4):
            snapshots = await asyncio.gather(
                *[store.get_account_identity(ctx.account_id) for _ in range(4)]
            )
            assert snapshots == [admitted] * 4

    async def management():
        for index in range(3):
            await store.create_account(f"parallel{index}", "admin")
            await store._reload()
            assert await store.get_account_identity(ctx.account_id) == admitted

    with QueueLoop() as queue:
        results = await asyncio.wait_for(
            asyncio.gather(
                service.resources.add_resource(
                    path=str(sample_markdown_file), ctx=ctx, reason="cross-loop", wait=True
                ),
                checks(),
                result(queue.submit(checks())),
                management(),
            ),
            30,
        )
    ingest = results[0]
    assert ingest.get("status") == "success", ingest
    assert ingest.get("root_uri"), ingest
    assert len(observed_loops) == 2
