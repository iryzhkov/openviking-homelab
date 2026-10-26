"""Official SDK over TCP; synthetic principal, real HTTP adapter and Reader."""

import asyncio
import hashlib
import inspect
import json
import os
import socket
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from importlib import import_module, metadata
from pathlib import Path

import httpx2
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from ov_http.app import create_app
from ov_http.auth import Principal
from ov_http.config import RESOURCES_PROFILE, RESOURCES_REVISION, RESOURCES_URI_POLICY, Config
from ov_http.document_write import DocumentWriter
from ov_http.reader import Reader

ROOT = "viking://resources/composite"
RAW_EXTERIOR = " \tα raw cobalt\r\n\nend \n"


class Receipt:
    """Persist every received value before later assertions."""

    def __init__(self, target):
        self.target = target
        self.value = {"state": "in_progress", "stage": "setup", "steps": [], "writes": []}
        self.save()

    def save(self):
        self.target.write_text(json.dumps(self.value, indent=2) + "\n")

    def step(self, name, **observation):
        self.value["stage"] = name
        self.value["steps"].append({"step": name, **observation})
        self.save()


def actual_origins():
    names = [
        "mcp",
        "mcp.client.session",
        "mcp.client.streamable_http",
        "ov_http.app",
        "ov_http.auth",
        "ov_http.config",
        "ov_http.document_write",
        "ov_http.mutations",
        "ov_http.reader",
    ]
    source = Path(os.environ["COMPOSITE_HTTP_SOURCE"]).resolve()
    rows = []
    for name in names:
        module = import_module(name)
        origin = module.__file__
        assert origin is not None, module.__name__
        path = Path(origin).resolve()
        if name.startswith("ov_http."):
            assert path == source.joinpath(*name.split(".")).with_suffix(".py")
        rows.append(
            {
                "module": module.__name__,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    for symbol in (ClientSession, streamable_http_client, Reader, DocumentWriter):
        actual = inspect.unwrap(symbol)
        origin = inspect.getsourcefile(actual)
        assert origin is not None, actual
        path = Path(origin).resolve()
        rows.append(
            {
                "symbol": symbol.__module__ + "." + symbol.__qualname__,
                "unwrapped": actual is not symbol,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return rows


@asynccontextmanager
async def serving(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield sock.getsockname()[1]
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)
        sock.close()


@asynccontextmanager
async def session(config, narrow=False):
    app = create_app(config)

    async def verify(token):
        assert token == ("narrow" if narrow else "synthetic")
        return Principal(
            "fixture",
            (ROOT + "/raw.md",) if narrow else ("viking://resources",),
            int(time.time()) + 300,
            "synthetic-digest",
            ("read", "write", "remove", "maintain"),
            RESOURCES_URI_POLICY,
        )

    app.verifier.verify = verify
    async with serving(app) as port:
        async with httpx2.AsyncClient(
            headers={
                "Host": "ov.example",
                "Authorization": "Bearer " + ("narrow" if narrow else "synthetic"),
            },
            trust_env=False,
            timeout=55,
        ) as client:
            async with streamable_http_client(
                f"http://127.0.0.1:{port}/mcp", http_client=client
            ) as streams:
                async with ClientSession(streams[0], streams[1]) as sdk:
                    await sdk.initialize()
                    yield sdk


async def main():
    receipt = Receipt(Path(os.environ["COMPOSITE_RESULT"]))
    try:
        assert os.environ["HOME"] == os.environ["TMPDIR"] + "/home"
        key = Path(os.environ["HOME"]) / "file-parent-backend.key"
        key.write_text("synthetic-backend-key")
        key.chmod(0o600)
        config = Config(
            "https://ov.example/mcp",
            "https://issuer.example/",
            "https://issuer.example/jwks",
            os.environ["COMPOSITE_BACKEND"] + "/api/v1",
            str(key),
            semantic_profile=RESOURCES_PROFILE,
            semantic_backend_revision=RESOURCES_REVISION,
            namespace_roots=("viking://resources",),
        )
        config = replace(config, mutation_profile="resources-document-mutations-v1")
        receipt.value["sdk"] = metadata.version("mcp")
        receipt.value["imported_origins"] = actual_origins()
        async with session(config) as sdk:
            response = await sdk.call_tool(
                "memory_write",
                {
                    "to": ROOT,
                    "filename": "raw.md",
                    "text": "standard raw cobalt",
                    "overwrite": True,
                    "verify": True,
                },
            )
            ack = json.loads(response.content[0].text)
            receipt.value.update(ack=ack, is_error=response.is_error)
            receipt.save()
            assert not response.is_error, ack
            assert ack["content_state"] == ack["outcome"] == "committed"
            assert ack["representation"] == "raw_file"
            assert ack["retry_safe"] is False
        receipt.value["state"] = "passed"
    except BaseException as failure:
        receipt.value.update(state="failed", failure_type=type(failure).__name__)
        raise
    finally:
        receipt.save()


if __name__ == "__main__":
    asyncio.run(main())
