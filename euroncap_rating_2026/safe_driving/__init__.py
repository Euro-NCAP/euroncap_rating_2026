# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import click
from euroncap_rating_2026.safe_driving.generate_template import generate_template
from euroncap_rating_2026.safe_driving.preprocess import preprocess
from euroncap_rating_2026.safe_driving.compute_score import compute_score


@click.group(name="safe_driving")
def safe_driving_cli():
    """Commands for domain safe_driving."""
    pass


safe_driving_cli.add_command(generate_template)
safe_driving_cli.add_command(preprocess)
safe_driving_cli.add_command(compute_score)
