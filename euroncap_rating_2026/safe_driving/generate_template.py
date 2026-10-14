# Copyright 2025-2026, Euro NCAP IVZW
# Created by IVEX NV (https://ivex.ai)
#
# Licensed under the Apache License 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import pandas as pd
import logging

from euroncap_rating_2026 import common
from euroncap_rating_2026 import config
from euroncap_rating_2026.safe_driving import integrity
import os
import shutil
import openpyxl
from importlib.resources import files
import click

logger = logging.getLogger(__name__)
settings = config.Settings()


@click.command()
@common.with_footer
def generate_template():
    """
    Generate a template file for safe driving.
    This function copies the master template registered for the running
    command line's protocol (`sd_template.xlsx` for `euroncap_rating_2026`;
    see integrity.pristine_template_path) to the current working directory.
    The template file serves as a starting point for safe driving analysis.
    Outputs:
        - A file named after the master template in the current working
          directory.
    """
    protocol = common.normalize_protocol(common.get_cli_protocol())
    template_path = integrity.pristine_template_path(protocol)
    template_name = os.path.basename(template_path)
    print(f"[Safe Driving] Generating template for safe driving...")
    dest_path = os.path.join(os.getcwd(), template_name)
    shutil.copyfile(template_path, dest_path)
    wb = openpyxl.load_workbook(dest_path)
    common.blank_computed_columns(wb, integrity.COMPUTED_SHEET_COLUMNS)
    common.add_version_sheet_to_wb(wb, common.CliCommand.GENERATE_TEMPLATE)
    # A registered master may already carry the marker; stamping it again
    # is a no-op, and keeps the five domains' generate-template alike.
    common.add_cli_protocol_row_to_version_sheet(wb)
    wb.save(dest_path)
    print(f"Template generated at {dest_path}")
