import hashlib
import json
import os
import sys
from pathlib import Path

def pytest_collection_finish(session):
    source = Path(os.environ["PROOF_SOURCE"])
    testroot = Path.cwd()
    rows = []
    assert len(session.items) == 9
    for item in session.items:
        origin = Path(item.module.__file__).resolve()
        relative = origin.relative_to(testroot)
        expected = source / relative
        assert origin.read_bytes() == expected.read_bytes(), (origin, expected)
        rows.append({"nodeid": item.nodeid, "module": item.module.__name__,
                     "origin": str(origin), "sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
                     "source": str(expected)})
    Path(os.environ["FILTER_ORIGIN_OUT"], "collected-origins.json").write_text(
        json.dumps(rows, indent=2) + "\n")


def pytest_sessionfinish(session, exitstatus):
    source = Path(os.environ["PROOF_SOURCE"])
    base = json.loads(Path(os.environ["FILTER_IMAGE_INPUTS"]).read_text())
    rows = []
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        origin = Path(filename).resolve()
        parts = origin.parts
        if "openviking" not in parts and "openviking_cli" not in parts:
            continue
        package = "openviking_cli" if "openviking_cli" in parts else "openviking"
        relative = Path(*parts[parts.index(package):])
        digest = hashlib.sha256(origin.read_bytes()).hexdigest()
        expected = source / relative
        if expected.is_file():
            assert digest == hashlib.sha256(expected.read_bytes()).hexdigest(), name
            kind = "checked-out-production"
        else:
            assert base.get(str(relative)) == digest, (name, relative)
            kind = "pinned-image-exception"
        rows.append({"module": name, "origin": str(origin), "relative": str(relative),
                     "sha256": digest, "kind": kind})
    assert rows
    Path(os.environ["FILTER_ORIGIN_OUT"], "imported-origins.json").write_text(
        json.dumps(rows, indent=2) + "\n")
