"""Fetch the pinned OpenViking image by digest and unpack its root filesystem.

No container runtime is available inside a Chaldea job, so the proof runs the
image's own files directly. This script pulls the manifest and every layer
anonymously from ghcr.io, checks each against its content digest, and unpacks
the layers in order (honouring OCI whiteouts) into a job-owned directory. It
writes an identity record that the report quotes.

Usage: python3 -I fetch_image.py OUT_DIR IDENTITY_JSON
"""

import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request

REPO = "volcengine/openviking"
DIGEST = "sha256:2ae2799dac1e606218d9ffeb06af3b65c8218b3556555620a369ed81b7c5aa1f"
# Receipt-backed expected dependency-image config. No runtime inspection.
EXPECTED_CONFIG = "sha256:56a10cc7c1d32307877a2669a6d42b2eecd787ae1d6faf8d9e80d7737fefdcfa"
ACCEPT = ", ".join([
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def _get(url: str, token: str, accept: str | None = None):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    if accept:
        req.add_header("Accept", accept)
    return urllib.request.urlopen(req, timeout=120)


def _download(url: str, token: str, dest: str, digest: str) -> None:
    algo, want = digest.split(":", 1)
    h = hashlib.new(algo)
    with _get(url, token) as resp, open(dest, "wb") as out:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            h.update(block)
            out.write(block)
    if h.hexdigest() != want:
        raise SystemExit(f"digest mismatch for {digest}: got {h.hexdigest()}")


def _remove(path: str) -> None:
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def _unpack(layer: str, rootfs: str, stats: dict) -> None:
    with tarfile.open(layer, "r:*") as tar:
        for member in tar:
            name = member.name.lstrip("/")
            base = os.path.basename(name)
            parent = os.path.dirname(name)
            if base == ".wh..wh..opq":
                target = os.path.join(rootfs, parent)
                if os.path.isdir(target) and not os.path.islink(target):
                    for entry in os.listdir(target):
                        _remove(os.path.join(target, entry))
                stats["opaque"] += 1
                continue
            if base.startswith(".wh."):
                _remove(os.path.join(rootfs, parent, base[4:]))
                stats["whiteouts"] += 1
                continue
            if member.ischr() or member.isblk() or member.isfifo():
                stats["skipped_devices"] += 1
                continue
            dest = os.path.join(rootfs, name)
            if os.path.lexists(dest) and not (member.isdir() and os.path.isdir(dest)):
                _remove(dest)
            try:
                tar.extract(member, rootfs, filter="tar")
                stats["members"] += 1
            except (tarfile.FilterError, OSError) as exc:
                stats["errors"] += 1
                if len(stats["error_samples"]) < 10:
                    stats["error_samples"].append(f"{name}: {exc}")


def main() -> None:
    out_dir, identity_path = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    with urllib.request.urlopen(
            f"https://ghcr.io/token?scope=repository:{REPO}:pull", timeout=60) as resp:
        token = json.load(resp)["token"]
    base = f"https://ghcr.io/v2/{REPO}"
    with _get(f"{base}/manifests/{DIGEST}", token, ACCEPT) as resp:
        raw = resp.read()
    if "sha256:" + hashlib.sha256(raw).hexdigest() != DIGEST:
        raise SystemExit("manifest digest mismatch")
    manifest = json.loads(raw)
    config_digest = manifest["config"]["digest"]
    if config_digest != EXPECTED_CONFIG:
        raise SystemExit(f"config digest {config_digest} does not match pinned expected config {EXPECTED_CONFIG}")
    config_path = os.path.join(out_dir, "config.json")
    _download(f"{base}/blobs/{config_digest}", token, config_path, config_digest)
    with open(config_path) as fh:
        config = json.load(fh)
    rootfs = os.path.join(out_dir, "rootfs")
    os.makedirs(rootfs, exist_ok=True)
    stats = {"members": 0, "whiteouts": 0, "opaque": 0, "skipped_devices": 0,
             "errors": 0, "error_samples": []}
    layers = []
    for i, layer in enumerate(manifest["layers"]):
        path = os.path.join(out_dir, f"layer-{i:02d}")
        _download(f"{base}/blobs/{layer['digest']}", token, path, layer["digest"])
        _unpack(path, rootfs, stats)
        os.unlink(path)
        layers.append({"digest": layer["digest"], "size": layer["size"],
                       "mediaType": layer["mediaType"]})
    identity = {
        "manifest_digest": DIGEST,
        "config_digest": config_digest,
        "config_matches_pinned_expected_config": True,
        "architecture": config.get("architecture"),
        "os": config.get("os"),
        "layers": layers,
        "unpack": stats,
        "seconds": round(time.time() - started, 1),
    }
    with open(identity_path, "w") as fh:
        json.dump(identity, fh, indent=2)
    if stats["errors"]:
        raise SystemExit("image unpack failed; see identity record")
    print(json.dumps({k: identity[k] for k in ("config_digest", "architecture", "unpack", "seconds")}))


if __name__ == "__main__":
    main()
