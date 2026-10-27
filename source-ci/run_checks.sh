#!/bin/sh
set -eu
jobroot="$(pwd -P)"
scratch="$jobroot/.cache/exact-filter"
export HOME="$scratch/tmp/home" TMPDIR="$scratch/tmp" UV_CACHE_DIR="$scratch/uv-cache"
export XDG_CONFIG_HOME="$HOME/.config" XDG_CACHE_HOME="$HOME/.cache" PYTHONDONTWRITEBYTECODE=1
[ "$HOME" = "$TMPDIR/home" ]
case "$HOME" in "$TMPDIR"/*) ;; *) exit 2;; esac
case "$TMPDIR" in /tmp*) exit 2;; esac
[ "$(stat -f -c %T "$jobroot")" != tmpfs ]
mkdir -p "$HOME" "$TMPDIR" "$jobroot/filter-out"
out="$jobroot/filter-out"
finish() {
 rc=$?; trap - 0
 date -u +%FT%TZ > "$out/ended.txt"
 [ -s "$out/summary.json" ] || printf '{"exit_code":%s,"phase":"setup"}\n' "$rc" > "$out/summary.json"
 tar -czf "$jobroot/filter-results.tar.gz" -C "$jobroot" filter-out
 exit "$rc"
}
trap finish 0
trap 'exit 124' TERM INT
if [ -e .git ]; then
 python3 -I source-ci/attest.py collect "$out"
else
 [ -n "${SOURCE_CI_PREPARED:-}" ]
 cp "$SOURCE_CI_PREPARED/checkout.json" "$SOURCE_CI_PREPARED/inventory.json" "$out/"
fi
python3 -I source-ci/attest.py verify "$out"
printf 'HOME=%s\nTMPDIR=%s\n' "$HOME" "$TMPDIR" > "$out/scratch.txt"
sh -n source-ci/run_checks.sh
python3 -I - <<'PY'
import ast,json,pathlib
for p in pathlib.Path("source-ci").glob("*.py"):
    ast.parse(p.read_text(),str(p))
json.loads(pathlib.Path("source-ci/native-source-manifest.json").read_text())
PY
uv --version > "$out/uv.txt"
date -u +%FT%TZ > "$out/started.txt"
files="openviking/storage/viking_vector_index_backend.py tests/filter_contract/test_resources_exact_delete_filter.py"
timeout -k 5 90 uv tool run --from ruff==0.15.5 ruff check $files > "$out/ruff.txt" 2>&1
timeout -k 5 90 uv tool run --from ruff==0.15.5 ruff format --check $files > "$out/format.txt" 2>&1
timeout -k 10 150 python3 -I source-ci/fetch_image.py "$scratch/image" "$out/image-identity.json"
rootfs="$scratch/image/rootfs"
img_sp="$rootfs/app/.venv/lib/python3.13/site-packages"
venv="$HOME/venv"
sp="$venv/lib/python3.13/site-packages"
mkdir -p "$venv/bin" "$sp" "$scratch/imglibs" "$HOME/test-run"
ln -s "$rootfs/usr/local/bin/python3.13" "$venv/bin/python"
printf 'home = %s\ninclude-system-site-packages = false\nversion = 3.13\n' "$rootfs/usr/local/bin" > "$venv/pyvenv.cfg"
for entry in "$img_sp"/* "$img_sp"/.[!.]*; do
 [ -e "$entry" ] || continue
 name=$(basename "$entry")
 case "$name" in openviking|openviking_cli) continue;; esac
 ln -s "$entry" "$sp/$name"
done
cp -a "$img_sp/openviking" "$sp/openviking"
mkdir -p "$sp/openviking_cli"
python3 -I - "$sp" > "$out/image-provided-inputs.json" <<'PY'
import hashlib,json,pathlib,sys
sp=pathlib.Path(sys.argv[1])
print(json.dumps({str(p.relative_to(sp)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((sp/"openviking").rglob("*")) if p.is_file()},indent=2))
PY
(cd "$sp" && sha256sum --quiet -c "$jobroot/source-ci/base-manifest.sha256")
for f in "$rootfs"/usr/lib/x86_64-linux-gnu/*; do
 b=$(basename "$f")
 [ -d "$f" ] && [ ! -L "$f" ] && continue
 case "$b" in
 libc.so*|libc-*|libm.so*|libmvec*|ld-linux*|libpthread*|libdl.so*|librt.so*|libresolv*|libutil.so*|libanl*|libnsl*|libnss_*|libBrokenLocale*|libthread_db*|libmemusage*|libpcprofile*|libc_malloc_debug*) continue;;
 esac
 ln -sf "$f" "$scratch/imglibs/$b"
done
export LD_LIBRARY_PATH="$rootfs/usr/local/lib:$scratch/imglibs"
python3 -I - "$jobroot" "$sp" "$HOME/test-run" > "$out/overlay.txt" <<'PY'
import hashlib,pathlib,shutil,sys
root,sp,testroot=map(pathlib.Path,sys.argv[1:])
for folder in ("openviking","openviking_cli"):
    for source in sorted((root/folder).rglob("*")):
        if source.is_file():
            target=sp/source.relative_to(root)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
            assert target.read_bytes()==source.read_bytes()
            print(hashlib.sha256(target.read_bytes()).hexdigest(),source.relative_to(root))
shutil.copytree(root/"tests",testroot/"tests")
shutil.copyfile(root/"pyproject.toml",testroot/"pyproject.toml")
shutil.copyfile(root/"source-ci/origin_plugin.py",testroot/"origin_plugin.py")
for source in sorted((root/"tests").rglob("*")):
    if source.is_file():
        assert source.read_bytes()==(testroot/source.relative_to(root)).read_bytes()
PY
timeout -k 10 120 uv pip install --quiet --no-deps --target "$scratch/testdeps" --python-version 3.13 --python-platform x86_64-unknown-linux-gnu pytest==9.0.2 pytest-asyncio==1.3.0 pytest-cov==6.0.0 coverage==7.6.10 iniconfig==2.3.0 pluggy==1.6.0 pygments==2.19.2 packaging==25.0 sqlalchemy==2.0.54 greenlet==3.3.2
python3 -I - "$scratch/testdeps" > "$out/distributions.json" <<'PY'
import importlib.metadata,json,sys
print(json.dumps(sorted([{"name":d.metadata["Name"],"version":d.version} for d in importlib.metadata.distributions(path=[sys.argv[1]])],key=lambda d:d["name"]),indent=2))
PY
"$venv/bin/python" --version > "$out/python.txt"
"$venv/bin/python" - "$jobroot" > "$out/workflow-validation.json" <<'PY'
import json,pathlib,sys,yaml
root=pathlib.Path(sys.argv[1])
m=json.loads((root/"source-ci/native-source-manifest.json").read_text())
w=yaml.safe_load((root/".github/workflows/source-checks.yml").read_text())
assert w.get("on",w.get(True))=={"workflow_dispatch":None}
assert w["permissions"]=={"contents":"read"}
assert len(w["jobs"])==1
job=w["jobs"]["source-checks"]
assert job["runs-on"]=="ubuntu-24.04" and job["timeout-minutes"]==10
steps=job["steps"]
assert {s["uses"] for s in steps if "uses" in s}=={k+"@"+v for k,v in m["action_pins"].items()}
assert steps[0]["with"]=={"persist-credentials":False,"fetch-depth":0,"submodules":False}
assert next(s["with"] for s in steps if s.get("uses","").startswith("astral-sh/setup-uv@"))=={"version":m["uv_version"],"enable-cache":False}
assert not any(k in job for k in ("services","container","environment"))
print(json.dumps({"workflow_dispatch_only":True,"contents_read_only":True,"pinned_actions":m["action_pins"]},indent=2))
PY
export PYTHONPATH="$scratch/testdeps" PROOF_OUT="$out" PROOF_SOURCE="$jobroot" FILTER_IMAGE_INPUTS="$out/image-provided-inputs.json"
cd "$HOME/test-run"
"$venv/bin/python" - "$sp" "$jobroot" > "$out/origin.txt" <<'PY'
import hashlib,importlib,pathlib,sys
sp,root=map(pathlib.Path,sys.argv[1:])
for name in ("openviking.storage.viking_vector_index_backend","openviking.storage.expr","openviking.storage.vectordb_adapters.base","openviking.storage.vectordb_adapters.local_adapter","openviking.server.identity","openviking_cli.session.user_id"):
    module=importlib.import_module(name)
    actual=pathlib.Path(module.__file__).resolve()
    relative=pathlib.Path(*name.split(".")).with_suffix(".py")
    assert actual==sp/relative
    assert actual.read_bytes()==(root/relative).read_bytes()
    print(name,actual,hashlib.sha256(actual.read_bytes()).hexdigest())
PY
"$venv/bin/python" - "$out" <<'PY'
import json,os,pathlib,subprocess,sys,xml.etree.ElementTree as ET
out=pathlib.Path(sys.argv[1])
assert os.environ["HOME"]==os.environ["TMPDIR"]+"/home"
summary={"boundary":"pure filters; optional explicit export; no store effects","modes":[]}
origins=[]
for mode in ("with-proof","without-proof"):
    target=out/mode
    target.mkdir()
    env=dict(os.environ,PROOF_OUT=str(target),FILTER_ORIGIN_OUT=str(target),PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    cmd=[sys.executable,"-m","pytest","--noconftest","-p","pytest_asyncio.plugin","-p","origin_plugin","-p","no:cacheprovider","-o","addopts=","-q","--junitxml="+str(target/"junit.xml"),"tests/filter_contract/test_resources_exact_delete_filter.py"]
    if mode=="without-proof":
        cmd=["env","-u","PROOF_OUT",*cmd]
    try:
        with (target/"pytest.txt").open("w") as stream:
            code=subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=120).returncode
    except subprocess.TimeoutExpired:
        code=124
    row={"mode":mode,"exit_code":code}
    if (target/"junit.xml").is_file():
        root=ET.parse(target/"junit.xml").getroot()
        row["counts"]={key:sum(int(s.attrib.get(key,0)) for s in root.findall("testsuite")) for key in ("tests","errors","failures","skipped")}
    exported=list(target.glob("filters-*.json"))
    row["artifact_exports"]=len(exported)
    if not code:
        assert row.get("counts")=={"tests":9,"errors":0,"failures":0,"skipped":0},row
        assert len(exported)==(9 if mode=="with-proof" else 0),row
    if (target/"collected-origins.json").is_file():
        origins.extend(json.loads((target/"collected-origins.json").read_text()))
    summary["modes"].append(row)
    print(json.dumps(row),flush=True)
    if code:
        print((target/"pytest.txt").read_text()[-12000:],flush=True)
summary["exit_code"]=0 if all(row["exit_code"]==0 for row in summary["modes"]) else 1
summary["counts"]={key:sum(row.get("counts",{}).get(key,0) for row in summary["modes"]) for key in ("tests","errors","failures","skipped")}
(out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
(out/"collected-origins.json").write_text(json.dumps(origins,indent=2)+"\n")
print(json.dumps(summary),flush=True)
sys.exit(summary["exit_code"])
PY
