"""One fresh ADMIN/unindexed RAW overwrite through the real adapter and owning API."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import time

import httpx
import pytest

from openviking.server.app import create_app, _initialize_runtime_state
from openviking.server.config import ServerConfig
from openviking.server.dependencies import set_service
from openviking.service.core import OpenVikingService
from openviking.service import document_write as dw
from openviking_cli.session.user_id import UserIdentifier
from tests.server.conftest import _install_fake_embedder, _install_fake_vlm

ROOT = "viking://resources/composite"
CODES = {"CONFLICT", "INTERNAL", "FAILED_PRECONDITION", "PERMISSION_DENIED",
         "INVALID_ARGUMENT", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "NOT_FOUND",
         "UNAUTHENTICATED", "ALREADY_EXISTS", "WRITE_CONFLICT"}
TYPES = {"OpenVikingError", "ConflictError", "PermissionDeniedError", "FailedPreconditionError",
         "ValueError", "TypeError", "KeyError", "RuntimeError", "OSError", "PermissionError",
         "FileNotFoundError", "TimeoutError", "CancelledError", "AssertionError", "ExceptionGroup"}


def error_type(error):
    name = type(error).__name__
    return name if name in TYPES else "OtherError"


async def test_fresh_admin_deferred_raw_write(monkeypatch):
    home = Path(os.environ["HOME"])
    assert str(home) == os.environ["TMPDIR"] + "/home"
    out = Path(os.environ["PROOF_OUT"])
    report = {"state": "running", "phase": "initialize", "conditional_posts": 0,
              "checkpoints": {"prepared": False, "backup": False, "publish": False},
              "prepare_entered": False, "prepare_returned": False,
              "seed_processing_admission": False, "setup_waits": 0, "mutation_waits": 0,
              "teardown": "pending"}

    def save():
        (out / "closed-proof.json").write_text(json.dumps(report, indent=2) + "\n")

    def project(response):
        row = {"http_status": response.status_code}
        try:
            value = response.json()
            error = value.get("error", {})
            code = error.get("code")
            if code is not None:
                row["error_code"] = code if code in CODES else "OTHER"
            details = error.get("details", {})
            for key in ("content_state", "processing_state"):
                if details.get(key) in {"not_started", "target_unchanged", "rolled_back",
                    "rollback_failed", "rollback_unavailable", "unknown", "committed",
                    "unobserved", "failed", "accepted"}:
                    row[key] = details[key]
        except (ValueError, TypeError, AttributeError):
            row["envelope"] = "unobserved"
        return row

    fake = _install_fake_embedder(monkeypatch)
    _install_fake_vlm(monkeypatch)
    svc = OpenVikingService(path=str(home / "slice-data"),
                           user=UserIdentifier.the_default_user("slice-bootstrap"))
    root_key = secrets.token_hex(32)
    config = ServerConfig(auth_mode="api_key", root_api_key=root_key)
    app = create_app(config=config, service=svc)
    set_service(svc)
    failure = None
    save()
    try:
        await asyncio.wait_for(_initialize_runtime_state(app, svc, config), 60)
        assert svc.viking_fs.query_embedder is None
        provider = svc.viking_fs._embedding_provider
        assert provider is not None
        report["normal_embedding"] = {"legacy_none": True, "provider_present": True}
        real_bind = provider.bind
        def bind(account_id):
            if report["phase"] == "conditional_write":
                assert account_id == "image-fixture"
                report.setdefault("private_embedding_accounts", []).append("image-fixture")
                save()
            return real_bind(account_id)
        monkeypatch.setattr(provider, "bind", bind)
        real_prepare, real_write = dw._prepare, dw.write_document

        async def prepare(*args, **kwargs):
            report["prepare_entered"] = True; save()
            try:
                answer = await real_prepare(*args, **kwargs)
            except BaseException as exc:
                report["prepare_exception_type"] = error_type(exc); save()
                raise
            report["prepare_returned"] = True; save()
            return answer

        async def write(service, request, ctx, *, fault=None):
            assert fault is None
            async def checkpoint(phase):
                if phase in report["checkpoints"]:
                    report["checkpoints"][phase] = True; save()
            try:
                return await real_write(service, request, ctx, fault=checkpoint)
            except BaseException as exc:
                report["native_exception_type"] = error_type(exc)
                if exc.__cause__ is not None:
                    report["native_cause_type"] = error_type(exc.__cause__)
                    if type(exc.__cause__) is AttributeError:
                        import linecache
                        leaf = exc.__cause__.__traceback__
                        while leaf.tb_next is not None:
                            leaf = leaf.tb_next
                        report["exact_attribute_error"] = True
                        report["cause_at_private_embed_lookup"] = (
                            leaf.tb_frame.f_code is dw._embed_private_records.__code__
                            and linecache.getline(dw.__file__, leaf.tb_lineno).strip() == "fs.query_embedder.embed,"
                        )
                        save()
                code = getattr(exc, "code", None)
                if code is not None:
                    report["native_error_code"] = code if code in CODES else "OTHER"
                save()
                raise

        monkeypatch.setattr(dw, "_prepare", prepare)
        monkeypatch.setattr(dw, "write_document", write)
        inner = httpx.ASGITransport(app=app, raise_app_exceptions=False)

        class ObservedTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                publication = request.method == "POST" and request.url.path == "/api/v1/resources/document-write"
                if publication:
                    report["conditional_posts"] += 1; save()
                    assert report["conditional_posts"] == 1
                response = await inner.handle_async_request(request)
                await response.aread()
                if publication:
                    report["conditional_response"] = project(response); save()
                return response

        async with httpx.AsyncClient(transport=ObservedTransport(), base_url="http://slice",
                                     timeout=40) as wire:
            async def api(method, path, key, payload=None):
                async with asyncio.timeout(40):
                    result = await wire.request(method, "/api/v1" + path,
                        headers={"X-API-Key": key}, json=payload)
                report["last_route_response"] = project(result); save()
                return result

            report["phase"] = "admin_create"; save()
            absent = await api("GET", "/admin/accounts/image-fixture", root_key)
            assert absent.status_code == 404
            created = await api("POST", "/admin/accounts", root_key,
                                {"account_id": "image-fixture", "admin_user_id": "operator"})
            assert created.status_code == 200
            tenant_key = created.json()["result"]["user_key"]
            identity = await app.state.api_key_manager.resolve_identity(tenant_key)
            report["identity"] = {"role": str(identity.role),
                                  "same_supported_key": True, "backend_generated": True}
            assert str(identity.role).lower() == "admin"
            assert identity.account_id == "image-fixture" and identity.user_id == "operator"
            save()

            async def settle(label):
                result = await api("POST", "/system/wait", root_key)
                assert result.status_code == 200
                queues = result.json()["result"]
                report.setdefault("waits", []).append({"label": label,
                    "error_counts": {k: v["error_count"] for k, v in queues.items()}})
                save()
                assert all(v["error_count"] == 0 for v in queues.values())

            report["phase"] = "mkdir"; save()
            for uri in (ROOT, ROOT + "/archive"):
                result = await api("POST", "/fs/mkdir", tenant_key, {"uri": uri})
                assert result.status_code == 200
                await settle("setup-mkdir")
                report["setup_waits"] += 1; save()

            report["phase"] = "deferred_seed"; save()
            result = await api("POST", "/content/batch-write", tenant_key,
                {"root_uri": ROOT, "defer_processing": True,
                 "operations": [{"uri": ROOT + "/raw.md", "content": "old raw\n",
                                 "mode": "create", "expected_absent": True}]})
            assert result.status_code == 200
            ack = result.json()["result"]
            assert ack["created"] == [ROOT + "/raw.md"] and not ack["updated"] and not ack["unchanged"]
            assert ack["semantic_status"] == ack["vector_status"] == "deferred"
            report["seed"] = {"created": 1, "semantic_status": "deferred", "vector_status": "deferred",
                              "processing_admission": False, "vector_injection": False}; save()

            source = Path(os.environ["PROOF_SOURCE"]) / "http-input"
            sys.path.insert(0, str(source))
            from ov_http.auth import Principal
            from ov_http.config import Config, RESOURCES_PROFILE, RESOURCES_REVISION, RESOURCES_URI_POLICY
            from ov_http.reader import Backend
            from ov_http.mutations import call_mutation
            import ov_http.document_write
            assert Path(ov_http.document_write.__file__).resolve() == source / "ov_http/document_write.py"
            assert hashlib.sha256(Path(ov_http.document_write.__file__).read_bytes()).hexdigest() == "e2c2311e1e89f29b994e2af8a050cde63831ba2d1ef85e68c915049b315e4903"
            keyfile = home / "tenant-key"; keyfile.write_text(tenant_key); keyfile.chmod(0o600)
            adapter_config = Config(resource="https://ov.example/mcp", issuer="https://fixture/",
                jwks_url="https://fixture/jwks", backend_url="http://slice/api/v1",
                backend_secret_file=str(keyfile), semantic_profile=RESOURCES_PROFILE,
                semantic_backend_revision=RESOURCES_REVISION, namespace_roots=("viking://resources",),
                mutation_profile="resources-document-mutations-v1")
            principal = Principal("synthetic", ("viking://resources",), int(time.time()) + 600,
                                  "synthetic", ("read", "write", "remove", "maintain"),
                                  RESOURCES_URI_POLICY)
            report["phase"] = "conditional_write"; save()
            async with httpx.AsyncClient(transport=ObservedTransport(),
                    headers={"X-API-Key": tenant_key}, timeout=35) as adapter_wire:
                backend = Backend(adapter_config, principal, adapter_wire)
                value, is_error = await call_mutation(backend, "memory_write",
                    {"to": ROOT, "filename": "raw.md", "text": "standard raw cobalt",
                     "verify": True, "overwrite": True}, {})
            report["adapter_result"] = {"is_error": bool(is_error),
                **{k: value.get(k) for k in ("content_state", "processing_state", "outcome", "retry_safe")}}
            save()
            assert report["conditional_posts"] == 1
            assert not is_error and value["content_state"] == "committed" and value["processing_state"] == "accepted"
            assert report.get("private_embedding_accounts") and all(x == "image-fixture" for x in report["private_embedding_accounts"])
            report["phase"] = "owning_wait"; save()
            await settle("memory_write")
            report["mutation_waits"] = 1
            report["state"] = "passed"; save()
    except BaseException as exc:
        failure = exc
        report["state"] = "failed"
        report["exception_type"] = error_type(exc)
        save()
    finally:
        try:
            await asyncio.wait_for(svc.close(), 35)
            plugin = getattr(app.state, "auth_plugin", None)
            if plugin is not None:
                await asyncio.wait_for(plugin.shutdown(), 5)
            report["teardown"] = "passed"
        except BaseException as exc:
            report["teardown"] = "failed"
            report["teardown_exception_type"] = error_type(exc)
            failure = failure or exc
        save()
    assert failure is None, "closed vertical-slice refusal; see closed-proof.json"
