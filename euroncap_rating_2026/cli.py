# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import click

from euroncap_rating_2026 import common
from euroncap_rating_2026.crash_avoidance import crash_avoidance_cli
from euroncap_rating_2026.crash_protection import crash_protection_cli
from euroncap_rating_2026.safe_driving import safe_driving_cli
from euroncap_rating_2026.post_crash import post_crash_cli
from euroncap_rating_2026.overall import overall_cli

from euroncap_rating_2026.config import logging_config
import logging

DOMAIN_COMMANDS = (
    crash_protection_cli,
    crash_avoidance_cli,
    safe_driving_cli,
    post_crash_cli,
    overall_cli,
)


def build_cli(protocol) -> click.Group:
    """One CLI group per rating protocol, sharing the same five domain
    sub-commands. The group records its protocol in the click context: each
    generate-template stamps it into the template's Version sheet
    (common.add_cli_protocol_row_to_version_sheet), and preprocess /
    compute-score refuse a file generated for another protocol
    (common.check_protocol). *protocol* must be registered
    (common.register_protocol); another programme's package builds its own
    command line by calling this with its protocol id.
    """
    protocol_id = common.normalize_protocol(protocol)
    programme = common.PROTOCOLS[protocol_id].programme_name

    @click.group(
        help=f"{programme} Rating Calculator 2026 application to compute NCAP scores.",
        context_settings=dict(
            help_option_names=["-h", "--help"],
        ),
    )
    @click.pass_context
    def group(ctx):
        """Main CLI entry point."""
        ctx.obj = {"protocol": protocol_id}

    for command in DOMAIN_COMMANDS:
        group.add_command(command)
    return group


logger = logging.getLogger(__name__)
logging_config()

# `euroncap_rating_2026` (console script) -- Euro NCAP 2026-2028.
cli = build_cli(common.DEFAULT_PROTOCOL)

if __name__ == "__main__":
    cli()
