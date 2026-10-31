#!/bin/sh
set -eu
jobroot="$(pwd -P)"
scratch="$jobroot/.cache/relations-native-regression-02"
export HOME="$scratch/tmp/home" TMPDIR="$scratch/tmp" UV_CACHE_DIR="$scratch/uv-cache"
export XDG_CONFIG_HOME="$HOME/.config" XDG_CACHE_HOME="$HOME/.cache" PYTHONDONTWRITEBYTECODE=1
[ "$HOME" = "$jobroot/.cache/relations-native-regression-02/tmp/home" ]
[ "$TMPDIR" = "$jobroot/.cache/relations-native-regression-02/tmp" ]
[ "$(stat -f -c %T "$jobroot")" != tmpfs ]
[ ! -e .git ]
umask 077
mkdir -p "$HOME" "$TMPDIR" "$jobroot/relations-out"
out="$jobroot/relations-out"
finish() {
 rc=$?; trap - 0
 printf '%s\n' "$rc" > "$out/job-exit-code.txt"
 date -u +%FT%TZ > "$out/ended.txt"
 tar -czf "$jobroot/relations-results.tar.gz" -C "$jobroot" relations-out
 exit "$rc"
}
trap finish 0
trap 'exit 124' TERM INT
date -u +%FT%TZ > "$out/started.txt"
printf 'HOME=%s\nTMPDIR=%s\n' "$HOME" "$TMPDIR" > "$out/scratch.txt"
uv --version > "$out/uv.txt"
python3 -I - <<'PY'
import ast,hashlib,json,pathlib
expected={
"openviking/storage/resource_rfv.py":"9b717ca45d1ec28656c0a529a8d91915030adbe2",
"openviking/service/document_write.py":"4dd84dcf279c2f860571a03e81419443696e124c",
"tests/server/test_document_write.py":"166406628eb1d39d177ea47c3feb749430e8f5d0",
"tests/server/test_reindex_effect_bounds_e2e.py":"2b1dd889fae1c62eae141c1e586f12a7c5d00bbd"}
rows={}
for name,blob in expected.items():
    data=pathlib.Path(name).read_bytes()
    assert hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()==blob,name
    ast.parse(data,filename=name)
    rows[name]={"git_blob":blob,"sha256":hashlib.sha256(data).hexdigest()}
pathlib.Path("relations-out/source-identity.json").write_text(json.dumps(rows,indent=2))
PY
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

export PYTHONPATH="$scratch/testdeps" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export REL_SOURCE="$jobroot" REL_OUT="$out" REL_SP="$sp"

python3 -I - "$sp" "$out" <<'PY'
import hashlib,json,pathlib,sys
sp,out=map(pathlib.Path,sys.argv[1:])
native={str(p.relative_to(sp)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sp.rglob("*") if p.is_file() and (p.suffix==".so" or "ragfs" in str(p.relative_to(sp)).lower())}
assert native["openviking/lib/ragfs_python.abi3.so"]=="ed576e8f1d57c31b2b3aa746ba0a108cb314667efa5898b6c072e65d63dd0484"
(out/"native-input-hashes.json").write_text(json.dumps(native,indent=2))
PY
"$venv/bin/python" - "$HOME/test-run/relations_runner.py" <<'PY'
import pathlib,sys
pathlib.Path(sys.argv[1]).write_text(r'''
import hashlib, importlib, importlib.metadata, json, os, pathlib, socket, sys
assert os.environ["HOME"] == os.environ["REL_SOURCE"] + "/.cache/relations-native-regression-02/tmp/home"
mode=sys.argv[1]
case=sys.argv[2]
root=pathlib.Path(os.environ["REL_SOURCE"])
sp=pathlib.Path(os.environ["REL_SP"])
out=pathlib.Path(os.environ["REL_OUT"])
def deny_connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError("external network forbidden in bounded source validation: " + repr(address))
    return original_connect(self,address)
original_connect=socket.socket.connect
socket.socket.connect=deny_connect
sources={}
for name in ("openviking.storage.resource_rfv","openviking.service.document_write"):
    m=importlib.import_module(name)
    p=pathlib.Path(m.__file__).resolve()
    rel=pathlib.Path(*name.split(".")).with_suffix(".py")
    assert p == sp / rel
    if mode=="current": assert p.read_bytes()==(root/rel).read_bytes()
    sources[name]={"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
(out/(case+"-origins.json")).write_text(json.dumps(sources,indent=2))
(out/"all-distributions.json").write_text(json.dumps(sorted([{"name":d.metadata["Name"],"version":d.version} for d in importlib.metadata.distributions()],key=lambda x:x["name"]),indent=2))
import pytest
import pytest_asyncio
import asyncio, gc, types
from openviking.storage.queuefs import QueueManager, get_queue_manager
from openviking.service.task_tracker import TaskTracker, get_task_tracker
original_enqueue=QueueManager.enqueue
original_has_work=TaskTracker.has_work
def assert_identities():
    assert QueueManager.enqueue is original_enqueue
    assert TaskTracker.has_work is original_has_work
    qm=get_queue_manager()
    tracker=get_task_tracker()
    assert qm.enqueue.__func__ is original_enqueue
    assert tracker.has_work.__func__ is original_has_work
    assert "enqueue" not in vars(qm) and "has_work" not in vars(tracker)
    print("REAL_QUEUE_TRACKER_IDENTITIES_PASS",flush=True)

class SafeFixtureEnvironment:
    test_started = False
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        self.test_started = True
        if "service" in item.fixturenames:
            assert_identities()
        yield
        if "service" in item.fixturenames:
            assert_identities()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        result=yield
        report=result.get_result()
        if case=="baseline" and call.when=="call" and call.excinfo:
            frames=call.excinfo.traceback
            local=next((entry.frame.f_locals for entry in frames
                        if entry.frame.code.name=="test_backup_sidecars_map_and_rollback_exact"),{})
            raised=local.get("raised")
            exc=raised.value if raised is not None else None
            chain=[]
            while exc is not None:
                chain.append({"type":type(exc).__name__,"message":str(exc),"details":getattr(exc,"details",None)})
                exc=exc.__cause__ or exc.__context__
            evidence={"checked":local.get("checked"),"chain":chain,"failure":str(call.excinfo.value)}
            (out/"baseline-failure.json").write_text(json.dumps(evidence,indent=2))

    @pytest.fixture(autouse=True)
    def bounded_fixture_environment(self, monkeypatch, tmp_path):
        self.test_started=False
        from tests.server import conftest as server_fixtures
        def omit_mock_only_fallback(service, monkeypatch):
            assert_identities()
            print("MOCK_ONLY_FALLBACK_OMITTED",flush=True)
        monkeypatch.setattr(server_fixtures,"_install_session_commit_queue_fallback",omit_mock_only_fallback)
        from openviking_cli.utils.config.open_viking_config import OpenVikingConfigSingleton
        config = {
            "storage":{"workspace":str(tmp_path/"workspace"),
                       "vectordb":{"name":"context","backend":"local"},
                       "agfs":{"backend":"local"}},
            "embedding":{"dense":{"provider":"openai","model":"stub-embed",
                         "api_key":"synthetic-fixture-key","api_base":"http://127.0.0.1:9/v1",
                         "dimension":2048,"input":"text"}},
            "vlm":{"provider":"openai","model":"stub-vlm","api_key":"synthetic-fixture-key",
                   "api_base":"http://127.0.0.1:9/v1","temperature":0.0,"max_retries":0},
            "encryption":{"enabled":False},
            "log":{"level":"WARNING","output":"stdout"}}
        assert str(tmp_path).startswith(os.environ["TMPDIR"]+"/")
        p=tmp_path/"ov.conf"
        p.write_text(json.dumps(config))
        monkeypatch.setenv("OPENVIKING_CONFIG_FILE",str(p))
        OpenVikingConfigSingleton.reset_instance()
        yield
        OpenVikingConfigSingleton.reset_instance()


    @pytest_asyncio.fixture(autouse=True)
    async def async_await_watchdog(self, bounded_fixture_environment):
        def dump():
            print("\nASYNC_AWAIT_CHAIN_BEGIN", flush=True)
            for task in asyncio.all_tasks():
                print("TASK", task.get_name(), repr(task), flush=True)
                coro=task.get_coro()
                seen=set()
                while coro is not None and id(coro) not in seen:
                    seen.add(id(coro))
                    frame=getattr(coro,"cr_frame",None) or getattr(coro,"gi_frame",None) or getattr(coro,"ag_frame",None)
                    if frame is not None:
                        safe={k:repr(v)[:250] for k,v in frame.f_locals.items() if k in ("uri","path","timeout","timeout_secs","queue_name","scope","attempt") and isinstance(v,(str,int,float,bool,type(None)))}
                        print(frame.f_code.co_filename,frame.f_lineno,frame.f_code.co_name,safe,flush=True)
                    else:
                        print(type(coro).__name__,repr(coro),flush=True)
                    nxt=getattr(coro,"cr_await",None) or getattr(coro,"gi_yieldfrom",None) or getattr(coro,"ag_await",None)
                    if nxt is None and type(coro).__name__=="async_generator_asend":
                        nxt=next((v for v in gc.get_referents(coro) if isinstance(v,types.AsyncGeneratorType)),None)
                    coro=nxt
            print("ASYNC_AWAIT_CHAIN_END",flush=True)
        async def watch():
            await asyncio.sleep(30)
            if not self.test_started:
                dump()
                print("INITIALIZE_30S_CAP_BLOCKED",flush=True)
                os._exit(42)
        watcher=asyncio.create_task(watch(), name="fixture-diagnostic-watchdog")
        yield
        watcher.cancel()

args=["-s","-vv","--tb=long","-p","pytest_asyncio.plugin","-p","no:cacheprovider","-o","addopts=",
*({"rollback":["tests/server/test_document_write.py::test_backup_sidecars_map_and_rollback_exact"],
"baseline":["tests/server/test_document_write.py::test_backup_sidecars_map_and_rollback_exact"],
"document":["tests/server/test_document_write.py"],
"reindex":["tests/server/test_reindex_effect_bounds_e2e.py"]}[case]),
"--junitxml="+str(out/(case+"-junit.xml"))]
sys.exit(pytest.main(args,plugins=[SafeFixtureEnvironment()]))
''')
PY
cd "$HOME/test-run"
run_case() {
 mode="$1"; label="$2"; cap="$3"
 set +e
 timeout -k 3 "$cap" "$venv/bin/python" relations_runner.py "$mode" "$label" > "$out/$label.txt" 2>&1
 result=$?
 set -e
 cat "$out/$label.txt"
 printf '%s\n' "$result" > "$out/$label-rc.txt"
 return "$result"
}
suite_rc=0
run_case current document 240 || suite_rc=$?
run_case current reindex 240 || suite_rc=$?
exit "$suite_rc"
