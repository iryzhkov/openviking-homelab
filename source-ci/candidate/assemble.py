"""Fixed file-only backend candidate recipe. Run only in disk-backed job scratch."""
import ast
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tarfile
import traceback
import urllib.request

SITE = "app/.venv/lib/python3.13/site-packages/"
NATIVE = SITE + "openviking/lib/ragfs_python.abi3.so"
CACHE = "var/empty/openviking-candidate-pycache"
DOCKER = "application/vnd.docker."


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(data):
    return "sha256:" + sha(data)


def encode(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def safe(name):
    p = PurePosixPath(name)
    assert name and not p.is_absolute() and str(p) == name
    assert ".." not in p.parts and "\\" not in name
    assert not any(x.startswith(".wh.") for x in p.parts)
    return name


def regular(path):
    for p in (path, *path.parents):
        assert not p.is_symlink(), p
    assert stat.S_ISREG(path.stat().st_mode), path
    return path.read_bytes()


def tar_bytes(tar, member):
    stream = tar.extractfile(member)
    assert stream is not None
    return stream.read()


def record(path, value):
    path.write_bytes(json.dumps(value, indent=2).encode() + b"\n")


def main():
    root = Path.cwd().resolve()
    scratch = root / ".cache/candidate"
    assert os.environ["TMPDIR"] == str(scratch)
    assert os.environ["HOME"] == str(scratch / "home")
    assert Path(os.environ["TMPDIR"]).resolve() == scratch
    assert Path(os.environ["HOME"]).resolve() == scratch / "home"
    assert Path(os.environ["HOME"]).parent == Path(os.environ["TMPDIR"])
    assert "/tmp" != str(scratch) and not scratch.is_symlink()
    assert subprocess.check_output(["stat", "-f", "-c", "%T", str(root)]).strip() != b"tmpfs"
    scratch.mkdir(parents=True, exist_ok=True)
    for name in ("home", "tmp", "evidence", "objects"):
        (scratch / name).mkdir(exist_ok=True)
    ev = scratch / "evidence"
    objects = scratch / "objects"
    recipe = root / "source-ci/candidate"
    fixed = json.loads(regular(recipe / "inputs.json"))
    selection_raw = regular(root / "selection.json")
    assert sha(selection_raw) == fixed["selection_sha256"]
    selection = json.loads(selection_raw)
    binding = json.loads(regular(root / "job-binding.json"))
    for row in binding["recipe_files"]:
        data = regular(root / safe(row["path"]))
        assert sha(data) == row["sha256"]
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == row["blob"]
    ast.parse(regular(recipe / "assemble.py"))
    assert selection["accepted_product"] == fixed["product"]
    assert selection["native_git_inputs"] == {
        k: v for k, v in binding["native_git_inputs"].items()
    }
    assert binding["native_git_inputs"]["crates/ragfs"] == "1e51698f7751655a43518bb8c5675db0c68768b7"
    assert binding["native_git_inputs"]["crates/ragfs-python"] == "bba6c803bc9ccd2e3c23553538f92487be4a7698"
    assert binding["native_git_inputs"]["Cargo.toml"] == "f44b1edc3eec9bec943a5e37148b2a832d1f6c17"
    assert binding["native_git_inputs"]["Cargo.lock"] == "585b8b6e4485ca1113e549f48e8205783d3c3112"
    assert binding["dependencies_unchanged_from_base_origin"] is True
    entries = {}
    product_rows = [x for x in selection["files"] if x["product"]]
    assert len(product_rows) == fixed["product_count"]
    assert sum(x["bytes"] for x in product_rows) == fixed["product_bytes"]
    for directory in ("openviking", "openviking_cli"):
        for path in (root / directory).rglob("*"):
            assert not path.is_symlink(), path
    actual_product = {
        p.relative_to(root).as_posix()
        for directory in ("openviking", "openviking_cli")
        for p in (root / directory).rglob("*") if not p.is_dir()
    }
    assert actual_product == {x["path"] for x in product_rows}
    for row in selection["files"]:
        path = safe(row["path"])
        data = regular(root / path)
        assert len(data) == row["bytes"] and sha(data) == row["sha256"]
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == row["blob"]
        mode = 0o755 if row["mode"] == "100755" else 0o644
        assert row["mode"] in ("100644", "100755")
        assert stat.S_IMODE((root / path).stat().st_mode) == mode
        dest = SITE + path if row["product"] else "usr/share/licenses/openviking-candidate/" + path
        assert dest not in entries
        entries[dest] = (data, mode)
    native = fixed["native"]
    assert json.loads(regular(root / "native-input.json")) == native
    assert native["sourceCommit"] == "0ff04af959195ea3dd5f3c0e4314fc2209fd0053"
    native_bytes = bytearray()
    for i, part in enumerate(native["parts"]):
        assert part["path"] == f"input/native-part-{i:02d}"
        data = regular(root / part["path"])
        assert sha(data) == part["sha256"] and len(data) == part["bytes"]
        native_bytes.extend(data)
    assert len(native["parts"]) == 9
    assert len(native_bytes) == native["bytes"] and sha(native_bytes) == native["sha256"]
    entries[NATIVE] = (bytes(native_bytes), 0o755)
    assert not set(entries).intersection(fixed["retained"])
    overlay_inventory = {
        k: {"sha256": sha(v[0]), "bytes": len(v[0]), "mode": oct(v[1])}
        for k, v in sorted(entries.items())
    }
    def layer(path):
        with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tar:
            for name, (data, mode) in sorted(entries.items()):
                safe(name)
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), mode
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(data))
    overlay = scratch / "overlay.tar"
    layer(overlay)
    layer(scratch / "overlay-again.tar")
    assert overlay.read_bytes() == (scratch / "overlay-again.tar").read_bytes()
    with tarfile.open(overlay) as tar:
        assert [m.name for m in tar] == sorted(entries)
        for member in tar.getmembers():
            data, mode = entries[member.name]
            assert member.isfile() and member.mode == mode
            assert member.uid == member.gid == member.mtime == 0
            assert not member.uname and not member.gname
            assert tar_bytes(tar, member) == data
    tool_archive = scratch / "crane.tar.gz"
    with urllib.request.urlopen(fixed["crane"]["url"], timeout=120) as response:
        tool_archive.write_bytes(response.read())
    assert tool_archive.stat().st_size == fixed["crane"]["bytes"]
    assert sha(tool_archive.read_bytes()) == fixed["crane"]["sha256"]
    crane = scratch / "crane"
    with tarfile.open(tool_archive) as tar:
        member = tar.getmember("crane")
        assert member.isfile()
        crane.write_bytes(tar_bytes(tar, member))
    crane.chmod(0o755)
    commands = []
    def run(*args):
        commands.append([str(crane), *map(str, args)])
        result = subprocess.run(commands[-1], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            sys.stderr.buffer.write(result.stderr)
            result.check_returncode()
        return result.stdout
    version = run("version").decode().strip()
    assert "0.22.1" in version
    for command in ("mutate", "validate", "digest", "config", "manifest"):
        (ev / f"help-{command}.txt").write_bytes(run(command, "--help"))
    (ev / "crane-version.txt").write_text(version + "\n")
    base_raw = run("manifest", fixed["base"])
    assert digest(base_raw) == fixed["base"].split("@")[1]
    base = json.loads(base_raw)
    base_config_raw = run("config", fixed["base"])
    if digest(base_config_raw) != fixed["base_config"]:
        base_config_raw = base_config_raw.removesuffix(b"\n")
    assert digest(base_config_raw) == fixed["base_config"]
    base_config = json.loads(base_config_raw)
    assert base["config"]["digest"] == fixed["base_config"]
    assert base_config["os"] == "linux" and base_config["architecture"] == "amd64"
    labels = {
        "org.opencontainers.image.revision": fixed["product"],
        "org.opencontainers.image.source": "https://github.com/iryzhkov/openviking-homelab",
        "org.opencontainers.image.version": "candidate-" + fixed["product"][:12],
        "io.openviking.candidate.tooling-tree": binding["candidate_tree"],
        "io.openviking.candidate.base-digest": fixed["base"].split("@")[1],
        "io.openviking.candidate.base-revision": fixed["base_origin"],
        "io.openviking.candidate.base-version": "0.5.2",
        "io.openviking.candidate.native-sha256": native["sha256"],
    }
    env = {"OPENVIKING_WITH_BOT": "0", "PYTHONPYCACHEPREFIX": "/" + CACHE, "PYTHONDONTWRITEBYTECODE": "1"}
    candidate = scratch / "candidate.docker.tar"
    args = ["mutate", fixed["base"], "--append", str(overlay), "--output", str(candidate)]
    for key, value in labels.items():
        args += ["--label", key + "=" + value]
    for key, value in env.items():
        args += ["--env", key + "=" + value]
    run(*args)
    (ev / "validate.txt").write_bytes(run("validate", "--tarball", str(candidate)))
    image_digest = run("digest", "--tarball", str(candidate)).decode().strip()
    descriptors = []
    final_base_files = {}
    ancestor_kinds = {}
    needed = set(entries) | set(fixed["retained"])
    ancestors = {str(p) for name in needed | {CACHE} for p in PurePosixPath(name).parents if str(p) != "."}
    base_cache_entries = []
    with tarfile.open(candidate) as saved:
        saved_members = saved.getmembers()
        assert all(m.isfile() for m in saved_members)
        assert len({m.name for m in saved_members}) == len(saved_members)
        save_raw = tar_bytes(saved, "manifest.json")
        save_manifest = json.loads(save_raw)
        assert len(save_manifest) == 1
        save = save_manifest[0]
        config_raw = tar_bytes(saved, save["Config"])
        config = json.loads(config_raw)
        assert len(save["Layers"]) == len(base["layers"]) + 1
        for i, name in enumerate(save["Layers"]):
            compressed = tar_bytes(saved, name)
            desc = {"mediaType": DOCKER + "image.rootfs.diff.tar.gzip", "size": len(compressed), "digest": digest(compressed)}
            descriptors.append(desc)
            raw = gzip.decompress(compressed)
            assert digest(raw) == config["rootfs"]["diff_ids"][i]
            if i == len(base["layers"]):
                assert raw == overlay.read_bytes()
                (objects / desc["digest"].split(":")[1]).write_bytes(compressed)
                continue
            assert desc["digest"] == base["layers"][i]["digest"]
            assert desc["size"] == base["layers"][i]["size"]
            with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
                for member in tar:
                    path = member.name.removeprefix("./").lstrip("/").rstrip("/")
                    if not path:
                        continue
                    assert not (path == CACHE or path.startswith(CACHE + "/"))
                    if path in ancestors:
                        ancestor_kinds[path] = member.isdir()
                    # Closed fixed-base inspection, no extraction or general whiteout handling.
                    if ".wh." in PurePosixPath(path).name:
                        parent = str(PurePosixPath(path).parent)
                        leaf = PurePosixPath(path).name
                        target = parent if leaf == ".wh..wh..opq" else parent + "/" + leaf[4:]
                        assert not any(p == target or p.startswith(target + "/") for p in needed | {CACHE})
                        continue
                    if path.startswith((SITE + "openviking/", SITE + "openviking_cli/")) and path.endswith(".pyc"):
                        assert "__pycache__" in PurePosixPath(path).parts
                        base_cache_entries.append(path)
                    if path in needed:
                        assert member.isfile(), path
                        final_base_files[path] = {"sha256": sha(tar_bytes(tar, member)), "mode": oct(member.mode), "bytes": member.size}
    assert all(ancestor_kinds.values())
    for path, expected in fixed["retained"].items():
        assert final_base_files[path]["sha256"] == expected
    assert final_base_files[NATIVE]["sha256"] == "ed576e8f1d57c31b2b3aa746ba0a108cb314667efa5898b6c072e65d63dd0484"
    assert config["rootfs"]["diff_ids"][:-1] == base_config["rootfs"]["diff_ids"]
    assert config["history"][:-1] == base_config["history"]
    old_other = {k: v for k, v in base_config.items() if k not in ("config", "rootfs", "history")}
    new_other = {k: v for k, v in config.items() if k not in ("config", "rootfs", "history")}
    assert new_other == old_other, (old_other, new_other)
    for key in set(base_config["config"]) | set(config["config"]):
        if key not in ("Env", "Labels"):
            assert base_config["config"].get(key) == config["config"].get(key), key
    expected_labels = dict(base_config["config"].get("Labels", {}), **labels)
    assert config["config"]["Labels"] == expected_labels
    expected_env = dict(x.split("=", 1) for x in base_config["config"]["Env"])
    expected_env.update(env)
    assert dict(x.split("=", 1) for x in config["config"]["Env"]) == expected_env
    assert len(config["config"]["Env"]) == len(expected_env)
    # Docker-save reload normalizes descriptor media types. Bind these exact bytes
    # to crane's authoritative --tarball digest before accepting the conversion.
    manifest = {"schemaVersion": 2, "mediaType": DOCKER + "distribution.manifest.v2+json",
                "config": {"mediaType": DOCKER + "container.image.v1+json", "size": len(config_raw), "digest": digest(config_raw)},
                "layers": descriptors}
    manifest_raw = encode(manifest)
    assert digest(manifest_raw) == image_digest, (digest(manifest_raw), image_digest)
    (objects / sha(config_raw)).write_bytes(config_raw)
    (objects / sha(manifest_raw)).write_bytes(manifest_raw)
    (ev / "base-manifest.json").write_bytes(base_raw)
    (ev / "base-config.json").write_bytes(base_config_raw)
    (ev / "docker-save-manifest.json").write_bytes(save_raw)
    record(ev / "overlay-files.json", overlay_inventory)
    record(ev / "replaced-base-files.json", final_base_files)
    record(ev / "base-descriptors.json", {"repository": fixed["base"].split("@")[0], "layers": base["layers"]})
    (ev / "selection.json").write_bytes(selection_raw)
    record(ev / "job-binding.json", binding)
    record(ev / "commands.json", commands)
    record(ev / "result.json", {
        "image_digest": image_digest, "config_digest": digest(config_raw),
        "layer_digest": descriptors[-1]["digest"], "layer_diff_id": digest(overlay.read_bytes()),
        "image_media_type": manifest["mediaType"], "config_media_type": manifest["config"]["mediaType"],
        "base_media_type": base["mediaType"], "base_layers": len(base["layers"]),
        "base_layer_bytes_external": sum(x["size"] for x in base["layers"]),
        "product_files": len(product_rows), "overlay_files": len(entries),
        "ignored_base_cache_entries": len(base_cache_entries), "cache_prefix_verified_absent": "/" + CACHE,
        "deterministic_overlay": True, "validate_exit": 0,
        "native_source_correspondence": binding["native_git_inputs"],
        "base_origin": "pinned-image evidence; no independent dependency/native/CLI rebuild",
        "runtime": "pending; file assembly only; no image execution or service startup",
        "crane_version": version, "crane_sha256": sha(crane.read_bytes()),
        "python_version": sys.version, "source_binding": binding,
        "media_conversion": "Docker-save reload uses Docker schema2/config/layer media types; compressed base bytes unchanged",
    })
    # Reconstruct another Docker-save from retained new objects and exact external
    # base blobs already present in the first save. No layer or config rebuild.
    reconstructed = scratch / "reconstructed.docker.tar"
    with tarfile.open(candidate) as original, tarfile.open(reconstructed, "w") as out:
        for name in [save["Config"], *save["Layers"], "manifest.json"]:
            if name == save["Config"]:
                data = regular(objects / sha(config_raw))
            elif name == save["Layers"][-1]:
                data = regular(objects / descriptors[-1]["digest"].split(":")[1])
            elif name == "manifest.json":
                data = regular(ev / "docker-save-manifest.json")
            else:
                data = tar_bytes(original, name)
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            out.addfile(info, io.BytesIO(data))
    run("validate", "--tarball", str(reconstructed))
    assert run("digest", "--tarball", str(reconstructed)).decode().strip() == image_digest
    record(ev / "commands.json", commands)
    custody = {str(p.relative_to(scratch)): {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
               for directory in (ev, objects) for p in sorted(directory.iterdir())}
    record(scratch / "custody.json", custody)
    bundle = root / "candidate-patch.tar"
    with tarfile.open(bundle, "w") as tar:
        for path in [scratch / "custody.json", *sorted(ev.iterdir()), *sorted(objects.iterdir())]:
            data = regular(path)
            info = tarfile.TarInfo(str(path.relative_to(scratch)))
            info.mode, info.size = 0o644, len(data)
            info.uid = info.gid = info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    assert bundle.stat().st_size <= 64 * 1024 * 1024
    print(json.dumps({"bundle_bytes": bundle.stat().st_size, "bundle_sha256": sha(bundle.read_bytes()),
                      "image_digest": image_digest, "reconstruction": "validate/digest passed"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure = traceback.format_exc()
        sys.stderr.write(failure)
        with tarfile.open("candidate-patch.tar", "w") as bundle:
            data = failure.encode()
            info = tarfile.TarInfo("FAILURE.txt")
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
        sys.exit(1)
