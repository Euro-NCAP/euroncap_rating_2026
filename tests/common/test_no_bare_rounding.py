# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Guard against bare rounding in the scoring code.

Python's round(), pandas' .round() and bare math.floor() diverge from
the protocol's Excel-equivalent decimal half-up rounding; every rounding
of a score or measured value must go through euroncap_rating_2026.numeric
(round_half_up / floor_score). This test greps the production package so
a new bare call fails CI with a pointer to the helpers.
"""

import re
import unittest
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[2] / "euroncap_rating_2026"

# (pattern, human-readable name, replacement)
FORBIDDEN = [
    (re.compile(r"(?<![\w.])round\("), "round(", "numeric.round_half_up"),
    (re.compile(r"math\.floor\("), "math.floor(", "numeric.floor_score"),
    (re.compile(r"\.round\("), ".round(", ".map(lambda v: round_half_up(v, n))"),
]

# The helper module itself is the one place bare rounding belongs.
EXCLUDED_FILES = {"numeric.py"}

# Cosmetic, non-score call sites that may keep bare rounding.
# file (relative to the package) -> allowed line contents
ALLOWLIST = {
    # RGB colour rendering for report cells — not score math
    "crash_protection/report_writer.py": ["int(round(c * 255))"],
    "crash_avoidance/report_writer.py": ["int(round(c * 255))"],
    "safe_driving/report_writer.py": ["int(round(c * 255))"],
}


def _is_allowlisted(rel_path, line):
    return any(fragment in line for fragment in ALLOWLIST.get(rel_path, []))


class TestNoBareRoundingInScoringCode(unittest.TestCase):
    def test_no_bare_round_or_floor(self):
        violations = []
        for py_file in sorted(PACKAGE_DIR.rglob("*.py")):
            rel_path = py_file.relative_to(PACKAGE_DIR).as_posix()
            if rel_path in EXCLUDED_FILES:
                continue
            for lineno, line in enumerate(py_file.read_text().splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for pattern, name, replacement in FORBIDDEN:
                    if pattern.search(line) and not _is_allowlisted(rel_path, line):
                        violations.append(
                            f"{rel_path}:{lineno}: bare {name} — use "
                            f"{replacement} instead: {stripped}"
                        )
        self.assertEqual(
            violations,
            [],
            "Bare rounding found in scoring code:\n" + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
