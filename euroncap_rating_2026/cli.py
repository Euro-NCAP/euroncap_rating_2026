# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import click

from euroncap_rating_2026.crash_avoidance import crash_avoidance_cli
from euroncap_rating_2026.crash_protection import crash_protection_cli
from euroncap_rating_2026.safe_driving import safe_driving_cli
from euroncap_rating_2026.post_crash import post_crash_cli
from euroncap_rating_2026.overall import overall_cli

from euroncap_rating_2026.config import logging_config
import logging


@click.group(
    help="Euro NCAP Rating Calculator 2026 application to compute NCAP scores.",
    context_settings=dict(
        help_option_names=["-h", "--help"],
    ),
)
def cli():
    """Main CLI entry point."""
    pass


logger = logging.getLogger(__name__)
logging_config()
cli.add_command(crash_protection_cli)
cli.add_command(crash_avoidance_cli)
cli.add_command(safe_driving_cli)
cli.add_command(post_crash_cli)
cli.add_command(overall_cli)

if __name__ == "__main__":
    cli()
