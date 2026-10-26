"""Pure filter contracts. No collection, index, filesystem or delete effects."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.expr import And, Eq, In, Or
from openviking.storage.vectordb_adapters.local_adapter import LocalCollectionAdapter
from openviking.storage.viking_vector_index_backend import VikingVectorIndexBackend
from openviking_cli.session.user_id import UserIdentifier

pytestmark = pytest.mark.asyncio


async def captured_filters(uris):
    ctx = RequestContext(user=UserIdentifier("account-exact", "caller"), role=Role.USER)
    filters = []

    async def capture_only(expr):
        filters.append(expr)
        return 0

    async def get_capture_only_backend(actual):
        assert actual is ctx
        return SimpleNamespace(delete_by_filter=capture_only)

    facade = SimpleNamespace(_get_backend_for_context=get_capture_only_backend)
    await VikingVectorIndexBackend.delete_uris(facade, ctx, uris)
    # Bypass construction: only the existing pure compiler is used.
    compiler = object.__new__(LocalCollectionAdapter)
    assert not vars(compiler)
    compiled = [compiler._compile_filter(expr) for expr in filters]
    target = Path(os.environ["PROOF_OUT"]) / "filters.json"
    evidence = json.loads(target.read_text()) if target.exists() else []
    evidence.append({"requested": uris, "compiled": compiled, "effects": "capture_only"})
    target.write_text(json.dumps(evidence, indent=2) + "\n")
    return ctx, filters, compiled


@pytest.mark.parametrize(
    "uri",
    [
        "viking://resources",
        "viking://resources/document.md",
        "viking://resources/team/document.md",
        "viking://resources/Case%20Name.md",
    ],
)
async def test_resource_entries_compile_at_depth_zero(uri):
    ctx, filters, compiled = await captured_filters([uri])
    assert filters == [
        And(
            [
                Eq("account_id", ctx.account_id),
                Or([Eq("uri", uri), Eq("uri", uri + "/")]),
            ]
        )
    ]
    # Existing adapter encoding strips a terminal slash; both remain depth zero.
    encoded = "/" + uri[len("viking://") :]
    assert compiled == [
        {
            "op": "and",
            "conds": [
                {"op": "must", "field": "account_id", "conds": [ctx.account_id]},
                {
                    "op": "or",
                    "conds": [
                        {"op": "must", "field": "uri", "conds": [encoded], "para": "-d=0"},
                        {"op": "must", "field": "uri", "conds": [encoded], "para": "-d=0"},
                    ],
                },
            ],
        }
    ]


@pytest.mark.parametrize(
    "uri",
    [
        "viking://user/alice/memories/item",
        "viking://agent/bot/skills/name",
        "viking://resources-other/item",
    ],
)
async def test_nonresource_legacy_filter_is_unchanged(uri):
    ctx, filters, compiled = await captured_filters([uri])
    legacy = And([Eq("account_id", ctx.account_id), Or([Eq("uri", uri), In("uri", [uri + "/"])])])
    assert filters == [legacy]
    compiler = object.__new__(LocalCollectionAdapter)
    assert compiled == [compiler._compile_filter(legacy)]
    clauses = compiled[0]["conds"][1]["conds"]
    assert clauses[0]["para"] == "-d=0"
    assert "para" not in clauses[1]


async def test_mixed_list_preserves_each_literal_member():
    uris = [
        "viking://resources/listed",
        "viking://user/alice/memories/listed",
        "viking://resources/listed/explicit-child",
    ]
    ctx, filters, compiled = await captured_filters(uris)
    assert len(filters) == len(uris) == len(compiled)
    for uri, expr in zip(uris, filters, strict=True):
        assert expr.conds[0] == Eq("account_id", ctx.account_id)
        scope = expr.conds[1]
        assert scope.conds[0] == Eq("uri", uri)
        expected = (
            Eq("uri", uri + "/")
            if uri.startswith("viking://resources/")
            else In("uri", [uri + "/"])
        )
        assert scope.conds[1] == expected


async def test_empty_list_requests_no_filter():
    _, filters, compiled = await captured_filters([])
    assert filters == compiled == []
