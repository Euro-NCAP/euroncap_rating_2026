# Euro NCAP Rating 2026

This repository provides tools for calculating Euro NCAP rating scores for 2026. It includes utilities for data conversion, score computation, and visualization of results.

## Installation

Euro NCAP Rating 2026 can be installed using pip.

### Prerequisites

Before installing the package, ensure you have the following prerequisites:

- **Python**: Version 3.10 or higher is required. You can download Python from the [official website](https://www.python.org/).

#### Using a Virtual Environment (Recommended)

It is recommended to use a Python virtual environment to isolate dependencies and avoid conflicts with other Python projects.

You only need to create the virtual environment once; after that, simply activate it whenever you work on the project.

**On Linux/macOS:**

```bash
python3 -m venv venv  # Create once
source venv/bin/activate  # Activate each time
```

**On Windows (Command Prompt):**

```cmd
python -m venv venv  # Create once
venv\Scripts\activate.bat  # Activate each time
```

**On Windows (PowerShell):**

```powershell
python -m venv venv  # Create once
venv\Scripts\Activate.ps1  # Activate each time
```

Once the virtual environment is activated, you can proceed with the installation steps below.

**From PyPi**

The recommended way to install the package is via PyPi, which is the default package index for Python.

To install the package from PyPi or upgrade to the latest version when a new release is available, use:

```bash
pip install --upgrade euroncap-rating-2026
```


**From GitHub repository**

It is also possible to install the package directly from the public GitHub repository or from a local clone.

Ensure you have the following additional prerequisites installed:

- **Git**: Required for installation from the GitHub repository. You can download Git from the [official website](https://git-scm.com/).

To upgrade to the latest version from the GitHub repository, use:

```bash
pip install --upgrade euroncap_rating_2026@git+https://github.com/Euro-NCAP/euroncap_rating_2026
```


## Usage

The application is organized into five domains, each exposed as a sub-command:

| Domain | Sub-command | Template prefix |
|--------|-------------|-----------------|
| Crash Protection | `crash_protection` | `cp_` |
| Crash Avoidance | `crash_avoidance` | `ca_` |
| Safe Driving | `safe_driving` | `sd_` |
| Post Crash | `post_crash` | `pc_` |
| Overall rating | `overall` | `overall_` |

The first four domains share the same three-step workflow. The `overall` domain has no `preprocess` step: it reads the four domain reports and computes the star rating.

**General workflow for a domain:**

1. Run `generate-template` to create the input template (`<prefix>template.xlsx`) in the current directory.
2. Fill in the required fields (grey cells) in the generated template.
3. Run `preprocess -i <prefix>template.xlsx` to expand the template (test points, load cases); this writes `<prefix>preprocessed_template.xlsx`.
4. Complete any additional required fields in the preprocessed file.
5. Run `compute-score -i <prefix>preprocessed_template.xlsx` to calculate the results; this writes `<prefix><date>_<time>_report.xlsx`.

### Example: Crash Protection

**Generate the input template:**

```bash
euroncap_rating_2026 crash_protection generate-template
```

This creates `cp_template.xlsx` in your current directory.

**Preprocess:**

After filling in the required fields (such as the VRU Prediction Matrix), run:

```bash
euroncap_rating_2026 crash_protection preprocess -i cp_template.xlsx
```

This generates `cp_preprocessed_template.xlsx` with the additional sheets needed for the assessment.

**Compute scores:**

```bash
euroncap_rating_2026 crash_protection compute-score -i cp_preprocessed_template.xlsx
```

### Example: Crash Avoidance, Safe Driving and Post Crash

The same three commands apply, with the domain name and template prefix changed:

```bash
euroncap_rating_2026 crash_avoidance generate-template
euroncap_rating_2026 crash_avoidance preprocess -i ca_template.xlsx
euroncap_rating_2026 crash_avoidance compute-score -i ca_preprocessed_template.xlsx

euroncap_rating_2026 safe_driving generate-template
euroncap_rating_2026 safe_driving preprocess -i sd_template.xlsx
euroncap_rating_2026 safe_driving compute-score -i sd_preprocessed_template.xlsx

euroncap_rating_2026 post_crash generate-template
euroncap_rating_2026 post_crash preprocess -i pc_template.xlsx
euroncap_rating_2026 post_crash compute-score -i pc_preprocessed_template.xlsx
```

### Example: Overall rating

The overall rating combines the four domain reports. Place the `*_report.xlsx` files produced by the four `compute-score` runs in one directory and pass it with `-p`:

```bash
euroncap_rating_2026 overall generate-template
euroncap_rating_2026 overall compute-score -i overall_template.xlsx -p <directory with the four *_report.xlsx files>
```

### Command-line Help

Every command and sub-command supports `--help`:

```bash
euroncap_rating_2026 --help
euroncap_rating_2026 crash_protection --help
euroncap_rating_2026 crash_protection compute-score --help
```

Each of the first four domains provides the sub-commands `generate-template`, `preprocess` and `compute-score`; `overall` provides `generate-template` and `compute-score`. Example:

```
$ euroncap_rating_2026 crash_protection --help
Usage: euroncap_rating_2026 crash_protection [OPTIONS] COMMAND [ARGS]...

  Commands for domain crash_protection.

Options:
  -h, --help  Show this message and exit.

Commands:
  compute-score      Compute NCAP scores from an input Excel file.
  generate-template  Generate a template for crash protection.
  preprocess         Preprocess VRU test points and generate loadcases...
```

`preprocess` and `compute-score` accept `-i/--input_file` (required) and `-o/--output_path` (defaults to the current directory).

## Version Handling

The application performs version checks at each processing step to ensure compatibility between different stages of the workflow.

Each command (`generate-template`, `preprocess`, and `compute-score`) writes its current version information to the output file. When a subsequent command reads an input file, it compares the version that created the file with the current application version.

**Version Mismatch Behavior:**

- **Patch version mismatch** (e.g., 1.0.0 vs 1.0.1): A warning is displayed, but processing continues. Patch updates typically contain bug fixes that don't affect compatibility.

- **Minor or major version mismatch** (e.g., 1.0.0 vs 1.1.0 or 2.0.0): Processing stops with an error message asking you to upgrade the application. Minor and major updates may include breaking changes or new features that require the latest version.

**Example:**

If you preprocessed a file with version 1.0.0 but are running `compute-score` with version 1.1.0, you'll see an error like:

```
Error: Version mismatch detected. File was created with version 1.0.0, but current version is 1.1.0.
Please upgrade the application using: pip install --upgrade euroncap-rating-2026
```

To resolve this, upgrade to the latest version:

```bash
pip install --upgrade euroncap-rating-2026
```

Then regenerate your files starting from the appropriate step.

## Input Format

The application expects the input file to be in `.xlsx` format.

- **Input Requirements**: Users must provide values for all cells in the template that are highlighted with a **light grey background**. These cells represent the required input data for the application to compute the scores.
- For the VRU test, the user must provide a prediction for each cell in the VRU Prediction Matrix by selecting a color-coded value. Each cell contains a dropdown menu with the available options, which represent the possible prediction outcomes. The selectable values are:

  - **Blue**
  - **Brown**
  - **Dark Red**
  - **Green**
  - **Green-20**
  - **Green-30**
  - **Green-40**
  - **Grey**
  - **Orange**
  - **Red**
  - **Yellow**



## Output Format

The output is an updated `.xlsx` file where all scoring cells are filled with computed scores.

The output file is saved with the naming convention `<prefix>DATE_TIME_report.xlsx`, where `<prefix>` is the domain prefix (`cp_`, `ca_`, `sd_`, `pc_` or `overall_`) and `DATE_TIME` is the current date and time in the format `YYYY-MM-DD_HH-MM-SS`. For example, a Crash Protection report generated on March 15, 2026, at 14:30:45 is named `cp_2026-03-15_14-30-45_report.xlsx`.

This naming convention ensures that each output file is unique and timestamped for easy identification.

- **Output Details**: The cells updated by the application are highlighted with a **yellow background** in the output file, making it easy to identify the computed results.



## Development

### Configuration Options

For development, different configuration options are available. The application can be run in debug mode, which provides additional logging and a GUI for debugging purposes.

The application supports two configuration options:

1. **`log_level`**: Controls the logging level of the application (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).


Configuration options can be specified using environment variables.

- `EURONCAP_RATING_2026_LOG_LEVEL`: Sets the logging level.

You can set the following environment variables before running the application:

**On Linux/macOS:**

```bash
export EURONCAP_RATING_2026_LOG_LEVEL=DEBUG
```

**On Windows (Command Prompt):**

```cmd
set EURONCAP_RATING_2026_LOG_LEVEL=DEBUG
```

**On Windows (PowerShell):**

```powershell
$env:EURONCAP_RATING_2026_LOG_LEVEL="DEBUG"
```


### Installation from source

To run tests and develop the project, you need to install it from source.

After cloning the repository, install the project using [Poetry](https://python-poetry.org/).

```bash
poetry install
```

After installing from source, the usage is similar to above.

```bash
Usage: euroncap_rating_2026 [OPTIONS] COMMAND [ARGS]...

  Euro NCAP Rating Calculator 2026 application to compute NCAP scores.

Options:
  -h, --help  Show this message and exit.

Commands:
  crash_avoidance   Commands for domain crash_avoidance.
  crash_protection  Commands for domain crash_protection.
  overall           Commands for domain overall.
  post_crash        Commands for domain post_crash.
  safe_driving      Commands for domain safe_driving.
```

## Tests

### Unit Tests

Unit tests can be executed with the command:

```bash
python -m unittest discover -s tests      # or: poetry run python -m unittest discover -s tests
```

The suite takes a couple of minutes and ends with a summary such as `Ran 1802 tests ... OK`. CI runs the suite on Linux and Windows for Python 3.10 to 3.13, checks formatting with `black --check`, and runs `generate-template` for all five domains as a smoke test (see `.github/workflows/test.yml`).

You can check more options for unittest at its [own documentation](https://docs.python.org/3/library/unittest.html).

## Python Library Licenses

Below is a list of the Python libraries used in this project along with their respective licenses and PyPI links.

| Library              | Version     | License       | PyPI Link                                      |
|----------------------|-------------|---------------|------------------------------------------------|
| pandas               | ^2.2.3      | BSD-3-Clause  | [pandas](https://pypi.org/project/pandas/)     |
| pydantic             | ^2.11.1     | MIT           | [pydantic](https://pypi.org/project/pydantic/) |
| pydantic-settings    | ^2.8.1      | MIT           | [pydantic-settings](https://pypi.org/project/pydantic-settings/) |
| openpyxl             | ^3.1.5      | MIT           | [openpyxl](https://pypi.org/project/openpyxl/) |
| click                | ^8.1.7      | BSD-3-Clause  | [click](https://pypi.org/project/click/)       |
