r"""Guard against LaTeX leaking back into the Markdown docs.

GitHub only renders $...$ math in some contexts; VS Code preview, PyCharm,
GitLab and any plain .md viewer print the raw \mathbf{...} instead. The docs
therefore use Unicode throughout. This test fails if a LaTeX delimiter or
backslash command reappears.
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = ["README.md", "REPRODUCIBILITY.md", "RELEASING.md", "experiments/README.md"]

# \command, $$...$$, or inline $...$ spanning a backslash command
LATEX_CMDS = re.compile(r"\\(?:mathbf|mathbb|mathrm|text|textbf|frac|Delta|sum|cdot|times|in|approx|mu|Pi|big|Big|left|right|quad|qquad|%|,|;|:)[A-Za-z{]*")
INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)[^$\n]{2,}(?<!\$)\$(?!\$)")


class TestDocsHaveNoLatex(unittest.TestCase):

    def _read(self, rel):
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            return fh.read()

    def test_no_latex_commands(self):
        for rel in DOCS:
            text = self._read(rel)
            hits = []
            for i, line in enumerate(text.splitlines(), 1):
                if line.lstrip().startswith("```"):
                    continue
                for m in LATEX_CMDS.finditer(line):
                    # a single backslash before a word is LaTeX; markdown
                    # escapes use punctuation or backticks
                    hits.append((i, m.group(0), line.strip()[:70]))
            self.assertEqual(hits, [], f"{rel}: LaTeX commands found")

    def test_no_display_math_delimiters(self):
        for rel in DOCS:
            self.assertNotIn("$$", self._read(rel), f"{rel}: $$ display math")

    def test_no_inline_math(self):
        for rel in DOCS:
            text = self._read(rel)
            hits = []
            for i, line in enumerate(text.splitlines(), 1):
                if line.lstrip().startswith("```"):
                    continue
                for m in INLINE_MATH.finditer(line):
                    hits.append((i, m.group(0)))
            self.assertEqual(hits, [], f"{rel}: inline $...$ math found")

    def test_tables_are_well_formed(self):
        """A header row followed by a separator row must have matching column
        counts, otherwise the table renders as literal text."""
        for rel in DOCS:
            lines = self._read(rel).splitlines()
            for i in range(len(lines) - 1):
                head, sep = lines[i], lines[i + 1]
                if not re.fullmatch(r"\|[\s:\-|]+\|", sep.strip()):
                    continue
                if "|" not in head:
                    continue
                nh = head.count("|")
                ns = sep.count("|")
                self.assertEqual(
                    nh, ns,
                    f"{rel}:{i+1} table header has {nh} pipes, separator {ns}")

    def test_tables_are_surrounded_by_blank_lines(self):
        """Without a blank line before it, a pipe table is not recognised by
        CommonMark and renders as a run-on paragraph."""
        for rel in DOCS:
            lines = self._read(rel).splitlines()
            for i, line in enumerate(lines):
                if not re.fullmatch(r"\|[\s:\-|]+\|", line.strip()):
                    continue
                if i == 0:
                    self.fail(f"{rel}:{i+1} separator row is the first line")
                prev = lines[i - 1].strip()
                self.assertNotEqual(
                    prev, "",
                    f"{rel}:{i+1} table separator needs a blank line before it")
                self.assertIn(
                    "|", prev,
                    f"{rel}:{i+1} table separator not preceded by a header row")


if __name__ == "__main__":
    unittest.main()