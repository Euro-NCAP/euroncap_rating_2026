# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# Single source of truth for overall's not-yet-computed columns, the
# sibling of the four domains' integrity.COMPUTED_SHEET_COLUMNS (overall
# has no trust boundary/HMAC, hence no integrity.py). generate_template
# blanks these to "-"; calculate_score reads them back as 0
# (numericize_computed_columns) so a missing domain report still yields
# 0 points / 0 stars.
COMPUTED_SHEET_COLUMNS = {
    "Rating": ["Star rating"],
    "Stage Scores": ["Score"],
    "Stage element Scores": ["Score"],
    "Test Scores": ["Score"],
}
