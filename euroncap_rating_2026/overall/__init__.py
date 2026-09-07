# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import click
from euroncap_rating_2026.overall.generate_template import generate_template
from euroncap_rating_2026.overall.compute_score import (
    compute_score as _compute_score_cmd,
    calculate_score,
)


@click.group(name="overall")
def overall_cli():
    """Commands for domain overall."""
    pass


overall_cli.add_command(generate_template)
overall_cli.add_command(_compute_score_cmd)
