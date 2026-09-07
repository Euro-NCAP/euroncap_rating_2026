# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import click
from euroncap_rating_2026.crash_avoidance.generate_template import generate_template
from euroncap_rating_2026.crash_avoidance.preprocess import preprocess
from euroncap_rating_2026.crash_avoidance.compute_score import compute_score


@click.group(name="crash_avoidance")
def crash_avoidance_cli():
    """Commands for domain crash_avoidance."""
    pass


crash_avoidance_cli.add_command(generate_template)
crash_avoidance_cli.add_command(preprocess)
crash_avoidance_cli.add_command(compute_score)
