# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import click
from euroncap_rating_2026.post_crash.generate_template import generate_template
from euroncap_rating_2026.post_crash.preprocess import preprocess
from euroncap_rating_2026.post_crash.compute_score import compute_score


@click.group(name="post_crash")
def post_crash_cli():
    """Commands for domain post_crash."""
    pass


post_crash_cli.add_command(generate_template)
post_crash_cli.add_command(preprocess)
post_crash_cli.add_command(compute_score)
