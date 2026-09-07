# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Protocol-conformant rounding primitives.

The Euro NCAP Overall Assessment protocol (v10.0, "Calculation rules")
defines explicit rounding stages (inputs 2 dp, criteria 2 dp, dummy 4 dp,
test 3 dp, stage floored) and the reference implementation is Excel, whose
``ROUND`` behaves as if applied to the 15-significant-digit decimal
rendering of the stored double — decimal half-up, immune to 1-ulp binary
artifacts. Python's ``round()`` (and pandas' ``.round()``) is round-half-even
on the exact binary value, which diverges from Excel both at decimal
midpoints (0.075 -> 0.07 instead of 0.08) and for values computed 1 ulp
below a midpoint or an integer.

Every rounding of a score or measured value in the scoring chain must go
through the helpers in this module instead of ``round()``/``math.floor()``;
``tests/common/test_no_bare_rounding.py`` enforces this.
"""

import math
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

#: Guard precision (decimal places) used to absorb binary noise before the
#: final rounding step. It is ~4 orders of magnitude finer than the finest
#: legitimate data resolution in the protocol (5 dp at body-region level),
#: so the guard quantize can only remove float artifacts — it can never
#: promote a value that is genuinely below a midpoint.
GUARD_DIGITS = 9


def round_half_up(value, ndigits, guard=GUARD_DIGITS):
    """Round ``value`` to ``ndigits`` decimals, half-up (Excel ``ROUND``).

    Two-stage quantize: the exact binary expansion of the float is first
    quantized at the guard precision to absorb 1-ulp artifacts (so a
    boundary computed as 0.4749999999999999 becomes exactly 0.475), then
    quantized to the target precision with decimal half-up tie-breaking
    (ties away from zero, matching Excel for negative values).

    ``None`` and NaN pass through unchanged so optional/pandas values can
    be rounded without special-casing at the call sites.
    """
    if value is None:
        return None
    if pd.isna(value):
        return value
    d = Decimal(float(value))
    d = d.quantize(Decimal(1).scaleb(-max(guard, ndigits + 4)), rounding=ROUND_HALF_UP)
    return float(d.quantize(Decimal(1).scaleb(-ndigits), rounding=ROUND_HALF_UP))


def floor_score(value):
    """Protocol floor for stage scores, immune to 1-ulp-below sums.

    Sums of legitimate 3-dp domain scores can land 1 ulp below the true
    integer (8.359 + 3.864 + 0.777 == 12.999999999999998); a bare
    ``math.floor`` then silently eats a whole point. Snapping at 6 dp
    first — three orders of magnitude finer than the 3-dp resolution of
    the summands — removes the artifact while a genuine 79.999 still
    floors to 79.
    """
    return math.floor(round_half_up(value, 6))
