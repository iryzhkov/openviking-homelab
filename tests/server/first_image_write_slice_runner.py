"""Run exactly one bounded native case; export closed regular evidence only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

out = Path(os.environ["PROOF_OUT"])
assert os.environ["HOME"] == os.environ["TMPDIR"] + "/home"
private = Path(os.environ["HOME"]) / "slice-private-output"
private.mkdir(mode=0o700)
case = out / "case-0"; case.mkdir()
name = "tests/server/test_first_image_write_vertical_slice.py::test_fresh_admin_deferred_raw_write"
env = dict(os.environ, PROOF_OUT=str(case))
command = [sys.executable, "-u", "-m", "pytest", "-p", "origin_plugin",
           "-q", "--tb=no", "--show-capture=no", "-p", "no:cacheprovider",
           "-o", "junit_logging=no", "-o", "junit_log_passing_tests=false",
           "--junitxml=" + str(private / "junit.xml"), name]
try:
    with (private / "case.log").open("w") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
    code = result.returncode
except subprocess.TimeoutExpired:
    code = 124
counts = {"tests": "0", "failures": "0", "errors": "1", "skipped": "0", "time": "0"}
if (private / "junit.xml").is_file():
    raw = ET.parse(private / "junit.xml").getroot()
    suites = list(raw) if raw.tag == "testsuites" else [raw]
    for key in ("tests", "failures", "errors", "skipped"):
        counts[key] = str(sum(int(s.get(key, "0")) for s in suites))
    counts["time"] = str(sum(float(s.get("time", "0")) for s in suites))
public = ET.Element("testsuites")
ET.SubElement(public, "testsuite", name="focused_native_vertical_slice", **counts)
(out / "junit.xml").write_bytes(ET.tostring(public))
summary = {"exit_code": code, "case": "fresh_admin_deferred_raw_write", "counts": counts,
           "raw_case_output_exported": False, "single_native_attempt": True}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(out / "pytest.txt").write_text(json.dumps(summary) + "\n")
if (case / "collected-origins.json").is_file():
    (out / "collected-origins.json").write_bytes((case / "collected-origins.json").read_bytes())
else:
    (out / "collected-origins.json").write_text("[]\n")
print(json.dumps(summary), flush=True)
if (case / "closed-proof.json").is_file():
    print((case / "closed-proof.json").read_text(), flush=True)
sys.exit(code)
