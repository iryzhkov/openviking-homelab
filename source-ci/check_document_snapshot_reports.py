"""Require the complete passing qualification suites, not only pytest exit zero."""

import json
import sys
from pathlib import Path
from xml.etree import ElementTree


def check_reports(directory):
    counts = {}
    for name, expected in (("document", 35), ("reindex", 30)):
        path = directory / (name + "-junit.xml")
        root = ElementTree.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        if not suites:
            raise ValueError(f"{name}: missing test suites")
        cases = [case for suite in suites for case in suite.iter("testcase")]
        if len(cases) != expected:
            raise ValueError(f"{name}: expected {expected} cases, found {len(cases)}")
        for suite in suites:
            if any(int(suite.get(key, "0")) for key in ("errors", "failures", "skipped")):
                raise ValueError(f"{name}: non-passing suite outcome")
        if any(
            case.find(tag) is not None for case in cases for tag in ("skipped", "failure", "error")
        ):
            raise ValueError(f"{name}: skipped, expected-failure or failed case")
        if sum(int(suite.get("tests", "0")) for suite in suites) != expected:
            raise ValueError(f"{name}: inconsistent suite count")
        counts[name] = expected
    return counts


if __name__ == "__main__":
    print(json.dumps({"passing_cases": check_reports(Path(sys.argv[1]))}, sort_keys=True))
