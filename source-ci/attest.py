"""Bind source-only checks to Git metadata, then verify source bytes without Git."""
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path.cwd()
MANIFEST = ROOT / "source-ci/native-source-manifest.json"
WORKFLOW = ".github/workflows/source-checks.yml"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args])


def text(*args):
    return git(*args).decode().strip()


def allowed(path):
    return path == WORKFLOW or path.startswith("source-ci/")


def safe(path):
    p = Path(path)
    assert not p.is_absolute() and ".." not in p.parts, path
    for parent in p.parents:
        assert not (ROOT / parent).is_symlink(), path
    return ROOT / p


def collect(out, prepared=None):
    m = json.loads(MANIFEST.read_text())
    base = m["accepted_backend_commit"]
    native = m["native_build_source_commit"]
    head = text("rev-parse", "HEAD")
    tree = prepared or text("rev-parse", "HEAD^{tree}")
    assert text("rev-parse", base + "^{tree}") == m["accepted_backend_tree"]
    for commit in (base, native):
        git("cat-file", "-e", commit + "^{commit}")
    git("merge-base", "--is-ancestor", base, head)
    git("merge-base", "--is-ancestor", native, base)
    changes = git("diff", "--name-status", "-z", "--no-renames", base, tree).split(b"\0")
    for status, path in zip(changes[0::2], changes[1::2]):
        assert status == b"A" and allowed(os.fsdecode(path)), (status, path)
    assert not git("diff", "--name-only"), "unstaged source changes"
    if not prepared:
        assert not git("diff", "--cached", "--name-only"), "staged source changes"
    assert not git("ls-files", "--others", "--exclude-standard"), "untracked inputs"
    correspondence = {}
    for path, expected in m["native_git_inputs"].items():
        observed = [text("rev-parse", ref + ":" + path) for ref in (tree, base, native)]
        assert observed == [expected] * 3, (path, observed)
        correspondence[path] = expected
    rows = []
    for entry in git("ls-tree", "-r", "-z", "--full-tree", tree).split(b"\0"):
        if not entry:
            continue
        info, rawpath = entry.split(b"\t", 1)
        mode, kind, blob = info.decode().split()
        path = os.fsdecode(rawpath)
        p = safe(path)
        st = p.lstat()
        assert kind == "blob", (path, kind)
        if mode == "120000":
            assert stat.S_ISLNK(st.st_mode), path
            data = os.fsencode(os.readlink(p))
            row = {"link_target": os.fsdecode(data)}
        else:
            assert mode in ("100644", "100755") and stat.S_ISREG(st.st_mode), path
            assert bool(st.st_mode & 0o111) == (mode == "100755"), path
            data = p.read_bytes()
            row = {}
        identity = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert identity == blob, path
        rows.append(dict(row, path=path, mode=mode, blob=blob, sha256=sha(data), size=len(data)))
    inventory = (json.dumps(rows, sort_keys=True, indent=2) + "\n").encode()
    workflow = next(row for row in rows if row["path"] == WORKFLOW)
    record = {
        "schema": "openviking-homelab/checkout/v1",
        "repository": "https://github.com/" + m["backend_repository"],
        "accepted_product_commit": base, "accepted_product_tree": m["accepted_backend_tree"],
        "checkout_commit": None if prepared else head, "checkout_tree": tree,
        "prepared_parent_commit": head if prepared else None,
        "identity_mode": "prepared-index-local-git" if prepared else "checkout-git",
        "hosted_git_gate": "pending" if prepared else "passed",
        "workflow_path": WORKFLOW, "workflow_blob": workflow["blob"],
        "workflow_sha256": workflow["sha256"], "workflow_sha": os.environ.get("GITHUB_WORKFLOW_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"), "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "run_url": None, "visibility_observation": "private; prior root publication receipt; not queried by CI",
        "inventory_sha256": sha(inventory), "tracked_count": len(rows),
        "regular_count": sum(row["mode"] != "120000" for row in rows),
        "symlink_count": sum(row["mode"] == "120000" for row in rows),
        "native_correspondence": correspondence, "native_build_is_ancestor": True,
        "allowed_additions": [WORKFLOW, "source-ci/**"], "product_unchanged": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner": {"system": platform.system(), "machine": platform.machine(), "libc": platform.libc_ver()},
        "action_pins": m["action_pins"], "uv_version": m["uv_version"], "ruff_version": m["ruff_version"],
        "limits": ["no native rehash or build", "no reproducible build", "no OCI candidate", "pure filter checks only"],
    }
    if record["run_id"]:
        record["run_url"] = record["repository"] + "/actions/runs/" + record["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "inventory.json").write_bytes(inventory)
    (out / "checkout.json").write_text(json.dumps(record, indent=2) + "\n")


def verify(out):
    m = json.loads(MANIFEST.read_text())
    record = json.loads((out / "checkout.json").read_text())
    raw = (out / "inventory.json").read_bytes()
    assert sha(raw) == record["inventory_sha256"]
    rows = json.loads(raw)
    checked = []
    for row in rows:
        path = row["path"]
        # A Chaldea selection contains only these paths. Hosted Git checks all files.
        selected = path.startswith(("openviking/", "openviking_cli/", "source-ci/")) or path in (
            "pyproject.toml", WORKFLOW, "tests/filter_contract/test_resources_exact_delete_filter.py"
        )
        if (ROOT / ".git").exists() or selected:
            p = safe(path)
            if row["mode"] == "120000":
                assert p.is_symlink(), path
                data = os.fsencode(os.readlink(p))
            else:
                assert stat.S_ISREG(p.lstat().st_mode), path
                data = p.read_bytes()
            assert sha(data) == row["sha256"], path
            checked.append(path)
    for path, expected in m["source_sha256"].items():
        assert sha(safe(path).read_bytes()) == expected, path
    assert sha((ROOT / "source-ci/base-manifest.sha256").read_bytes()) == m["base_manifest_sha256"]
    (out / "input-verification.json").write_text(json.dumps({
        "inventory_sha256": sha(raw), "verified_paths": checked,
        "mode": "full-checkout" if (ROOT / ".git").exists() else "selected-source-bytes",
        "hosted_git_gate": record["hosted_git_gate"],
        "execution_runner": {"system": platform.system(), "machine": platform.machine(), "libc": platform.libc_ver()},
    }, indent=2) + "\n")
    print(json.dumps({"verified_source_files": len(checked), "inventory_sha256": sha(raw)}))


if __name__ == "__main__":
    action, destination, *extra = sys.argv[1:]
    assert action in ("collect", "verify")
    if action == "collect":
        collect(Path(destination), extra[0] if extra else None)
    else:
        verify(Path(destination))
