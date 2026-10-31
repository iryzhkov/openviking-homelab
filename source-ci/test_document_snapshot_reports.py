"""Acceptance tests for the CI report gate."""

import os
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from check_document_snapshot_reports import check_reports


class ReportGateTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(self.scratch.cleanup)
        self.directory = Path(self.scratch.name)
        for name, count in (("document", 21), ("reindex", 30)):
            root = ElementTree.Element("testsuites")
            suite = ElementTree.SubElement(
                root, "testsuite", tests=str(count), errors="0", failures="0", skipped="0"
            )
            for index in range(count):
                ElementTree.SubElement(suite, "testcase", name=f"case-{index}")
            ElementTree.ElementTree(root).write(self.directory / (name + "-junit.xml"))

    def test_complete_passing_reports(self):
        self.assertEqual(check_reports(self.directory), {"document": 21, "reindex": 30})

    def test_nonpassing_cases_refuse_even_with_zero_summary(self):
        for tag, attributes in (
            ("skipped", {}),
            ("skipped", {"type": "pytest.xfail"}),
            ("failure", {}),
            ("error", {}),
        ):
            with self.subTest(tag=tag, attributes=attributes):
                path = self.directory / "document-junit.xml"
                root = ElementTree.parse(path)
                case = root.find(".//testcase")
                assert case is not None
                child = ElementTree.SubElement(case, tag, attrib=attributes)
                root.write(path)
                with self.assertRaises(ValueError):
                    check_reports(self.directory)
                case.remove(child)
                root.write(path)

    def test_twenty_document_cases_refuse(self):
        path = self.directory / "document-junit.xml"
        tree = ElementTree.parse(path)
        suite = tree.find("testsuite")
        assert suite is not None
        case = suite.find("testcase")
        assert case is not None
        suite.remove(case)
        suite.set("tests", "20")
        tree.write(path)
        with self.assertRaisesRegex(ValueError, "document: expected 21 cases, found 20"):
            check_reports(self.directory)

    def test_nonzero_summary_refuses(self):
        path = self.directory / "reindex-junit.xml"
        tree = ElementTree.parse(path)
        suite = tree.find("testsuite")
        assert suite is not None
        suite.set("errors", "1")
        tree.write(path)
        with self.assertRaises(ValueError):
            check_reports(self.directory)

    def test_missing_report_refuses(self):
        (self.directory / "document-junit.xml").unlink()
        with self.assertRaises(FileNotFoundError):
            check_reports(self.directory)
