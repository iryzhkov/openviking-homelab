# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""The mutation protocol probe performs no storage or Task operations."""


async def test_resource_mutation_contract_is_exact_and_storage_independent(client, monkeypatch):
    from openviking.server.routers import filesystem

    def unexpected_service_call():
        raise AssertionError("Protocol metadata must not inspect storage or Tasks")

    monkeypatch.setattr(filesystem, "get_service", unexpected_service_call)
    response = await client.get("/api/v1/fs/resources-only/mutation-contract")
    assert response.status_code == 200, response.text
    envelope = response.json()
    assert envelope["status"] == "ok"
    assert envelope["result"] == {
        "contract": "resources-document-mutations",
        "version": 1,
    }
