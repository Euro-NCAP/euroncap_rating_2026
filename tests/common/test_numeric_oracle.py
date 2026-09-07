# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Exact-arithmetic oracle for the guarded float rounding.

Property test: the production float path (float arithmetic + guarded
round_half_up at each protocol stage) must agree with a pure-Fraction
reference that never touches floats and resolves half-up ties in integer
arithmetic. Randomized protocol-shaped chains: 2-dp inputs, weights
including 1/3 and 5/24, quantized at 5 dp (body region), 4 dp (dummy)
and 3 dp (test score) — the protocol's precision ladder.

The seed is fixed so CI is deterministic; bump N locally for deeper runs.
"""

import random
import unittest
from fractions import Fraction

from euroncap_rating_2026.numeric import round_half_up

N_CHAINS = 10_000
SEED = 1457

# Protocol-shaped weights: exact thirds and the Sled & VT virtual
# loadcase weight 5/24 are the interesting (non-terminating) ones.
WEIGHTS = [
    Fraction(1, 3),
    Fraction(2, 3),
    Fraction(5, 24),
    Fraction(1, 4),
    Fraction(3, 4),
    Fraction(1, 1),
]


def exact_round_half_up(value: Fraction, ndigits: int) -> Fraction:
    """Reference rounding: pure integer arithmetic, ties away from zero."""
    scale = 10**ndigits
    num = value.numerator * scale
    den = value.denominator
    q, r = divmod(abs(num), den)
    if 2 * r >= den:
        q += 1
    if num < 0:
        q = -q
    return Fraction(q, scale)


class TestGuardedRoundingMatchesExactOracle(unittest.TestCase):
    def test_protocol_shaped_chains(self):
        rng = random.Random(SEED)
        mismatches = []
        for chain_index in range(N_CHAINS):
            n_criteria = rng.randint(2, 6)
            # 2-dp inputs in [0, 100], as both float and exact Fraction
            raw_cents = [rng.randint(0, 10_000) for _ in range(n_criteria)]
            weights = [rng.choice(WEIGHTS) for _ in range(n_criteria)]

            # --- float path (mirrors production arithmetic) ---
            body_f = [
                round_half_up((cents / 100) * (w.numerator / w.denominator), 5)
                for cents, w in zip(raw_cents, weights)
            ]
            dummy_f = 0.0
            for score in body_f:  # chained additions, like production sums
                dummy_f += score
            dummy_f = round_half_up(dummy_f, 4)
            test_f = round_half_up(dummy_f, 3)

            # --- exact path (no floats anywhere) ---
            body_q = [
                exact_round_half_up(Fraction(cents, 100) * w, 5)
                for cents, w in zip(raw_cents, weights)
            ]
            dummy_q = exact_round_half_up(sum(body_q), 4)
            test_q = exact_round_half_up(dummy_q, 3)

            for stage, got, expected in (
                ("body", body_f, [float(b) for b in body_q]),
                ("dummy", dummy_f, float(dummy_q)),
                ("test", test_f, float(test_q)),
            ):
                if got != expected:
                    mismatches.append(
                        (chain_index, stage, raw_cents, weights, got, expected)
                    )
                    break

        self.assertEqual(
            mismatches,
            [],
            f"{len(mismatches)} chains diverged from the exact oracle; "
            f"first: {mismatches[:1]}",
        )


if __name__ == "__main__":
    unittest.main()
