# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import click
from euroncap_rating_2026.crash_protection.generate_template import generate_template
from euroncap_rating_2026.crash_protection.preprocess import preprocess
from euroncap_rating_2026.crash_protection.compute_score import compute_score


@click.group(name="crash_protection")
def crash_protection_cli():
    """Commands for domain crash_protection."""
    pass


crash_protection_cli.add_command(generate_template)
crash_protection_cli.add_command(preprocess)
crash_protection_cli.add_command(compute_score)
