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
    This function copies the `sd_template.xlsx` file from the predefined
    data directory to the current working directory. The template file
    serves as a starting point for safe driving analysis.
    Outputs:
        - A file named `sd_template.xlsx` in the current working directory.
    """
    template_path = str(files("data").joinpath("sd_template.xlsx"))
    print(f"[Safe Driving] Generating template for safe driving...")
    dest_path = os.path.join(os.getcwd(), "sd_template.xlsx")
    shutil.copyfile(template_path, dest_path)
    wb = openpyxl.load_workbook(dest_path)
    common.blank_computed_columns(wb, integrity.COMPUTED_SHEET_COLUMNS)
    common.add_version_sheet_to_wb(wb, common.CliCommand.GENERATE_TEMPLATE)
    wb.save(dest_path)
    print(f"Template generated at {dest_path}")
