# Changelog

All notable changes to `euroncap-rating-2026` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow the package version in `pyproject.toml`, which every command stamps into
the workbooks it writes (a patch mismatch warns, a minor or major mismatch
rejects the file).

## [5.4.7] — 2026-09-07

Maintenance release preparing the repository for wider publication. No scoring
change: identical inputs produce identical scores and reports.

- Uniform copyright header (2025-2026) across all Python files; package author
  set to IVEX NV.
- Added CONTRIBUTORS.md and this changelog.
- README updated for all five domains, current command names and install
  instructions; `pdoc` moved to the development dependencies; repository links
  added to the package metadata.
- Code base formatted with black 25.1.0; formatting enforced in CI and
  pre-commit.
- Test code cleaned up.

## [5.4.6] — 2026-09-04

### Crash Avoidance
- Fixed 30 km/h green threshold for the CPLA/CBLA ACC scoring.
- CPLA/CBLA extended-range cells resolved from row attributes; collapse-and-
  promote rule for duplicate AEB rows at 50/60 km/h.
- Relaxed attribute matching for ACC verification rows.
- ELK RE under LDW: extended-range verification rows expect the DTLE at the
  warning instant; a Green prediction there is reported as inconsistent.
- Measured Orange against a predicted Brown now counts as correct.
- 2 km/h impact-speed tolerance (Frontal Collisions 4.2.4) applied to the
  relative impact speed only.
- Scenarios with no verification rows fall back to the predicted score instead
  of scoring 0.
- CCFhos verification limited to the pinned test point; CCFhol has none.

### Crash Protection
- VRU blue-point adjacency and headform selection fixes; blue count and tested
  bonus exposed through the VRU headform KPI accessor.
- Criteria gated off by a head Ares below 80 g clear their colour and
  Prediction.Check.

### Safe Driving
- Cut-out rows restored to the protocol's VUT/target speeds; superseded
  layouts are reported by the consistency check and still score identically.
- An all-Red ACC scenario scores 0.0 instead of staying unassessed.
- Blank verification values inherit the prediction colour consistently.

### Templates and I/O
- Not-yet-computed score cells are written as "-" in generated and
  preprocessed templates; "-" reads as unassessed everywhere.
- N/A handling harmonised across all templates; pandas no longer swallows the
  literal "N/A".
- Consistency checks limited to prediction sheets; dash-to-float conversion
  fixed; capped post-crash categories resolved when blanks cannot change the
  score.

## [5.4.5] — 2026-08-10

- Template modification (grey-cell integrity) check with signed hidden sheets.
- Missing-required-input and consistency-finding checks exposed as one list
  per domain, including a template version check.
- Grey input cells and dropdowns for the AEB/FCW function choice; AEB
  prevention rule for CBLA/CPLA.
- Half-up decimal rounding everywhere scores or colours are compared, removing
  1-ulp floating-point flips.
- ACC reference matrix, CP template labels and VRU blue-point selection fixed.
- "-" written instead of a PASS default for unfilled verification cells.

## [5.4.4] — 2026-07-02

- Driver Monitoring: selection, scoring cascade, grey/red cell rules and
  random test-scenario selection.
- Crash Avoidance: dark-green scoring, LSC denominator and cell weights,
  extended-range score, CPMFC scoring, CBDA verification rows, robustness
  layer with partner tests, verification-row iteration.
- Crash Protection: legform score with blank inspection, VRU criteria
  classification, blue-form ST/SA/T logic, whiplash and farside colours.
- Post Crash: eCall dependency rule, occupant extrication without input
  parameter dependency.
- Overall domain Python interface.
- Fixed ACC criteria and templates.

## [5.4.3] — 2026-05-29

- Robustness column in the Crash Avoidance preprocessed template.
- DataFrame-in/DataFrame-out interfaces for adding sheets and calculating
  scores.
- Manual test-point selection for Safe Driving and Crash Protection; ELK CC
  test points above 80 km/h filtered.
- VRU verification sheet generation and post-crash missing-sheet scoring fixed;
  matrix plots disabled by default.

## [5.4.2] — 2026-05-04

- Manual test-point selection option and selected-points interface.
- LDC tests filtering removed.

## [5.4.1] — 2026-04-23

- Target approach attribute on crossing scenarios.
- My extension capping removed for the front passenger in Frontal Offset.
- CCRb filtering lowered; template formats fixed.

## [5.4.0] — 2026-03-31

- SLIF N/A handling and Crash Protection farside fixes. Minor bump: files
  generated with 5.3.x are rejected.

## Earlier releases

- **5.3.0** — 2026-03-19 — Whiplash and CCRb colour fixes; Crash Avoidance
  template update.
- **5.2.0** — 2026-03-13 — Grey default input cells in all templates, Driver
  Monitoring fix, overall score maximised, Post Crash template update.
- **5.1.0** — 2026-03-10 — Safe Driving template update.
- **5.0.0** — 2026-02-20 — `overall` domain: star rating from the four domain
  reports.
- **4.0.0** — 2026-02-17 — `post_crash` domain.
- **3.0.0** — 2026-02-13 — `safe_driving` template, Version sheet stamped into
  every workbook, file/DataFrame scoring split, CBDA expected values and
  duplicated rows, pole-contact modifier, CCRb rows moved to AEB.
- **2.0.0 – 2.0.3** — 2025-11 — CPLA/CBLA handling, Crash Avoidance template
  update, version stamping and output naming.
- **1.0.0 – 1.1.0** — 2025-05 to 2025-07 — First release: Crash Protection
  and Crash Avoidance calculators, `generate-template` command, click CLI.
