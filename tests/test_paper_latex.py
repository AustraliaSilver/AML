"""Structural guards for the LaTeX paper.

There is no LaTeX toolchain in CI, so a broken `\\ref` or a dropped `\\end`
would only surface when someone compiles the paper by hand -- which is exactly
how the dead `export.py` / `test_export.py` references survived into the
appendix until they were found by hand. These checks are pure string analysis
and run in under a millisecond.

Escaping matters for the brace counter: `\\{` and `\\}` are literal delimiters,
so counting them as grouping braces reports a phantom imbalance. The scanner
therefore consumes the character *after* every backslash rather than only
skipping the backslash itself.
"""
import collections
import pathlib
import re
import unittest

PAPER = pathlib.Path(__file__).resolve().parent.parent / "paper_and_documents" / "aml_paper_v2_clean.tex"

REF_COMMANDS = ("ref", "eqref", "autoref", "pageref", "cref", "Cref", "nameref")


def strip_comments(text: str) -> str:
    """Drop `%` comments, keeping escaped `\\%`."""
    return re.sub(r"(?<!\\)%.*", "", text)


def strip_verbatim(text: str) -> str:
    """Blank out lstlisting/verbatim bodies, where braces are literal."""
    def blank(m):
        return re.sub(r"[^\n]", " ", m.group(0))
    return re.sub(r"\\begin\{(lstlisting|verbatim|minted)\}.*?\\end\{\1\}",
                  blank, text, flags=re.S)


def brace_balance(text: str) -> int:
    """Unclosed grouping braces, ignoring escaped `\\{` and `\\}`."""
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":          # consume the backslash AND its escaped char
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    return depth


class TestPaperLatex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw = PAPER.read_text(encoding="utf-8")
        cls.text = strip_comments(strip_verbatim(cls.raw))

    def test_paper_exists_and_is_not_empty(self):
        self.assertGreater(len(self.raw), 50_000)

    def test_every_ref_resolves_to_a_label(self):
        labels = set(re.findall(r"\\label\{([^}]*)\}", self.text))
        refs = set()
        for cmd in REF_COMMANDS:
            refs |= set(re.findall(r"\\" + cmd + r"\{([^}]*)\}", self.text))
        dangling = refs - labels
        self.assertFalse(dangling, "dangling references: %s" % sorted(dangling))

    def test_labels_are_unique(self):
        labels = re.findall(r"\\label\{([^}]*)\}", self.text)
        dupes = [k for k, v in collections.Counter(labels).items() if v > 1]
        self.assertFalse(dupes, "duplicate labels: %s" % sorted(dupes))

    def test_environments_are_paired(self):
        beg = collections.Counter(re.findall(r"\\begin\{([^}]*)\}", self.text))
        end = collections.Counter(re.findall(r"\\end\{([^}]*)\}", self.text))
        mismatched = {k: (beg[k], end[k]) for k in set(beg) | set(end) if beg[k] != end[k]}
        self.assertFalse(mismatched, "unbalanced environments: %s" % mismatched)

    def test_braces_balance_outside_verbatim(self):
        self.assertEqual(brace_balance(self.text), 0,
                         "unbalanced grouping braces outside verbatim blocks")

    def test_no_dangling_section_pointers(self):
        """`see Section~\\ref{...}` with a missing space or a stray tilde is the
        most common copy-paste defect and is invisible to the compiler until
        proofing."""
        self.assertNotRegex(self.text, r"Section~[^~\\]")


if __name__ == "__main__":
    unittest.main()