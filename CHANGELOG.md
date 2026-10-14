# Changelog

All notable changes to `euroncap-rating-2026` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow the package version in `pyproject.toml`, which every command stamps into
the workbooks it writes (a patch mismatch warns, a minor or major mismatch
rejects the file).

## [5.4.9] — 2026-10-14

### Fixed

- Crash protection: Rear Impact Static-Front is now scored per seat, Driver
  and Front Passenger at 0.5 points each, each from its own head-restraint
  readings. Before, one front seat was picked for the whole point (the one
  with the lower sum of criterion scores, the Driver on a tie), so the score
  could ignore a failing seat and depended on the seat order. Scores change
  when the two front seats differ (e.g. Driver all green, Front Passenger one
  green and three red: 1.0 -> 0.5). `CP - Dummy Scores` and
  `CP - Body region scores` now have a `Driver` and a `Front Passenger` row
  instead of the `Front` row, so the rows below move down by one. Files with
  the single `Front` row are still accepted: it reads as the Driver row and
  its `Inspection [%]` applies to both seats.
- Crash protection: a criterion with a single pass/fail limit (HPL equal to
  LPL) is now red, score 0, when its value is exactly at the limit; it was
  green. This can lower a score for Backset, Itibia, Ftibia, Dabdomen / Dchest
  compression, My,extension, Fz,tension and farside HIC15 / Ares-3ms values
  sitting exactly at the limit. Values below the limit score as before.
- Crash protection: the `Input parameters` sheet is now matched to the
  official template row by row label (Stage / Stage element / Stage
  subelement / Input parameter) instead of by position. Euro NCAP's Rear
  Impact Lab Assessment Sheet, which has no Frontal/Side placeholder rows and
  an extra notes column, is now accepted and read into our template's layout.
  A row with an unknown label, or the same row twice, is still rejected. A
  required row missing from the file is reported by the missing-input check
  with an empty cell reference. Files in our layout score exactly as before.
- Crash avoidance: the FC - Ped & Cyc report no longer fails, or puts its
  selected-point X marks two rows off, when its prediction frame was prepared
  before the CPLA/CBLA AEB-duplicate collapse existed. The grid is now only
  collapsed when the frame is collapsed too. Reports built from current
  frames are unchanged; no score changes.

- Crash avoidance: the Trajectory/Heading robustness layer now issues the
  target rotations of FC v1.2 §5.2.1.3 — -10° or +20° for CCRs, CMRs, CPNA,
  CPFA and CBNAO, +20° for CPNCO and CBNA, -20° for CBFA — instead of ±20° for
  every scenario. Only newly preprocessed files are affected; files that
  already carry a ±20° condition score as before.
- Safe driving, Driver Monitoring (protocol-driven score change, Step 7): a
  Warning that is only predicted Red no longer takes away the Long-distraction
  Intervention points. When a claimed Warning fails its verification test, a
  verified Intervention now scores by its own result, and an unverified one
  loses Forward and Lane support (Long distraction) or Lane support only (Short
  distraction and, now also, Phone use). Re-score affected files.
- Crash protection, VRU legform: when the test points are supplied with both a
  Femur and a Knee & Tibia point at the same column, preprocess no longer adds
  a second Knee & Tibia copy of that point from the Femur row. An SA-coloured
  copy was left unresolved and scored 0 in the Knee & Tibia result. Only newly
  preprocessed files are affected; re-preprocess affected campaigns.
- Safe driving, Driver Monitoring (protocol-driven score change): a FAIL on a
  Microsleep, Sleep or Unresponsive Intervention verification row now scores
  0 for that intervention. Before, the row was
  not found and counted as a pass. Warning, transient and Impairment rows are
  unchanged. Re-score affected files.
- Crash protection, whiplash: the Neck average now uses only the scoring
  criteria (NIC, +Fx, Fz), as Rear Impact v1.2 §3.1.2 describes. A blank
  capping-only row, e.g. Seatback deflection on Rear-High, used to count as 0
  in the average ((80+80+100+0)/4 = 65 % instead of 86.67 %). Capping is
  unchanged: a seatback deflection of 32° or more still caps the pulse.
  Whiplash tests with a blank capping-only row score higher; re-score them.
- Crash avoidance: the robustness layer scores 0 when no layer is claimed,
  whether its verification passes or fails. A failed verification used to
  give a negative Robustness score. Scores with one or more claimed layers are
  unchanged.

### Changed

- Safe driving: `DE - DM verif.` has one row per Driver Monitoring General
  requirement (14 rows, B2:B15 / I2:I15) instead of a single
  `General requirements` row, so the test-point table header moves from row 4
  to row 17 and B1 reads `Scenario`. Rows 2–8 take PASS/FAIL (N/A leaves the
  requirement unassessed, with a warning); rows 9–15 (non-functional system
  information) take PASS/FAIL/N/A, and N/A counts as a pass. Any blank still
  leaves Driver Monitoring unassessed and any FAIL still scores it 0. N/A is
  now kept in this sheet's `Value` column when the file is read
  (`data_model.PRESERVE_NA_SHEETS`); a DataFrame-in caller that reads the file
  itself must keep it too, otherwise N/A reads as blank. Files preprocessed
  before this change score the same.

## [5.4.8] — 2026-09-25

Protocol-driven scoring fixes across all four domains; re-score affected
campaigns. Templates, preprocessed files and reports written by 5.4.7 are still
accepted (a patch version difference only warns). The CLI commands and options
and the report sheets and columns are unchanged.

### Added

- Rating protocol registry, for other rating programmes built on this package:
  `common.Protocol`, `ProtocolSpec`, `register_protocol`, `resolve_protocol`
  and `check_protocol`, a `Protocol` row on the `Version` sheet,
  `cli.build_cli`, `speed_assistance.SpeedAssistanceProfile` /
  `register_profile` and `safe_driving.integrity.register_pristine_template`.
  Several public functions take a new optional `protocol` argument; omitting
  it keeps the previous behaviour. `euroncap_rating_2026` writes no `Protocol`
  row, and a workbook without one is Euro NCAP 2026-2028, so Euro NCAP
  templates and scores are unchanged. `preprocess`, `compute-score` and the
  `overall` roll-up refuse a file generated for another protocol.

- Crash protection, Rear Impact: the static rear assessment follows the Lab
  Assessment Sheet v1.3. The rear seat row is now `Rear (second row)`, a
  `Rear (third row)` block with its own `Torso angle (rear third row)` input
  parameter (a number, or `N/A` when the vehicle has no third row) was added
  on `CP - Rear Whiplash`, `CP - Dummy Scores` and `CP - Body region scores`,
  and the Static-Rear `Effective height modifier` row is replaced by the
  `Non-Use position` criterion (`PASS`/`FAIL`, or `1000`/`0`). Templates and
  preprocessed files produced by 5.4.5–5.4.7 are still accepted: their rows
  are matched under the previous labels, Static-Rear readings carry over, the
  value under the old modifier row is dropped (it was a height in mm, not a
  pass/fail flag) and the third row is treated as absent. New consistency
  check `rear_torso_angle` reports a non-numeric second-row angle or a
  third-row angle that is neither a number nor `N/A`. New
  `crash_protection.integrity.to_current_identity` and
  `legacy_identity_spellings` let consumers translate row identities written
  under the previous labels.

- Crash avoidance, CBDA (LSC dooring): new `Type of system` column on the
  `LSC - Ped & Cyc verif.` sheet, next to `Door selected`. It records which
  dooring function (Information / Warning / Retention) actually acted on that
  test point and is written by the tool, not by the lab: preprocess emits it
  blank, without input styling or a dropdown, the downstream evaluation tool
  (not this package) fills it from the measured dooring KPI on the same pass
  as `Value`, and compute-score reads it per row. It replaces `Function`,
  which is no longer written on CBDA rows
  (`Function` keeps its meaning for CPLA/CBLA). A blank `Type of system`
  falls back to the workbook-level `Vehicle response` with a warning.

### Changed

- Crash protection, Side Pole: the WorldSID-50 driver head `Ares` capping at
  80 g is now ignored when HIC15 stays below 500 -- a peak `Ares` with a low
  HIC15 comes from the arm hitting the head, not from the head hitting the
  pole, so the dummy is no longer capped to 0. The gate applies to the
  `Pole-32` load case only: the AE-MDB (`AEMDB-60`) and far side
  (`Main-AEMDB`, `Main-Pole`) `Ares` capping is unchanged, and the
  `Direct contact with pole` inspection input still caps the dummy regardless
  of HIC15. A missing HIC15 also still caps. The threshold is not in any
  protocol document yet and lives as `POLE_ARES_CAPPING_HIC15_GATE` in
  `crash_protection.dummy`. `Dummy.get_dummy_from_row` takes a new optional
  `load_case_name` argument; omitting it keeps the previous behaviour. This
  changes scores: an affected side pole test can only go up, never down.

- Crash avoidance, `v_rel_impact` colour bands (CCRs, CCRm, CCRb, CMRs, CMRb,
  CPNA, CPFA, CPNCO, CBNA, CBFA, CBNAO and the AEB rows of CPLA / CBLA): a
  measured impact speed exactly on a band edge now belongs to the band
  *above* (on the 60 km/h table 10.0 km/h is Orange, 20.0 Brown, 30.0 Red,
  where they used to be Yellow, Orange and Brown), the same convention as the
  Crash Protection criteria; Green stays the single point v = 0. The 2 km/h
  tolerance window of FC v1.2 §4.2.4 follows the same convention: for a
  middle band the accepted range is `edge − 2 ≤ v < edge + 2` (Yellow
  `0 < v < 12`, Orange `8 ≤ v < 22`, Brown `18 ≤ v < 32`), Green keeps
  `v < 2`, and Red is no longer widened downward (`v ≥ 30`, where a Red
  prediction used to be accepted from 28 km/h). Scoring change: any point whose
  `v_rel_impact` lies on a band edge or within 2 km/h of one can change colour
  or prediction result and should be re-scored. The default band convention
  of `matrix_processing.get_band_color` / `check_thresholds` is unchanged for
  callers that do not pass the new `half_open` / `degenerate_first` flags.

- Crash protection, Rear Impact scoring: Static-Rear splits its point per
  seat row into 75 % seat geometry and 25 % non-use position, with no non-use
  credit when the geometry scores 0 (Rear Impact protocol §3.1.3.4 /
  §3.1.4). Without a third row the second row carries the whole point; with
  one, the two rows carry 0.5 each. Files with no `Non-Use position` entry
  (including every file produced before this version) now top out at 0.75 of
  the rear point instead of 1.0, and the 790 mm front-seat height modifier is
  no longer applied to the rear row, where any reading below 790 mm used to
  zero the whole rear score. Each rear seat row's backset limit is computed
  from its own torso angle. The Farside input parameter is now spelled
  `Red line >125 mm outboard of the orange line ?` as in the official sheet;
  the previous spelling is still read.

- Crash avoidance, CBDA (LSC dooring) prediction consistency: the rule is now
  equal-or-worse instead of an allow-list of exact colours. A predicted colour
  is consistent when it is no better than the best colour the claimed
  `Vehicle response` can reach (Information → Brown, Warning → Yellow,
  Retention → Green); Grey is skipped and an unfilled matrix still passes.
  Predictions that are *worse* than the claim are now accepted (Brown under a
  Warning claim, Orange under Retention), where they were previously rejected.
  The check also runs over the full matrix and on both the random and the
  input-points selection branches, so the blocking preprocess check and the
  report-only consistency finding agree.

- Crash avoidance, CBDA scoring: the computed colour is now capped by the door
  the row was actually tested on, instead of trusting the predicted colour. A
  row passes its band from its own `Type of system` (Information: TTC ≥ 2.30;
  Warning: TTC ≥ 1.70; Retention: outside −0.40 < TTC < 1.70; a missing TTC
  always fails), then takes the worse of the door ceiling (Retention: Green
  off the driver's door, Yellow on it; Warning: Yellow off, Orange on;
  Information: Brown) and the OEM prediction, so a row never scores better
  than predicted. A missing `Door selected` is scored as the driver's door
  with a warning. This changes scores for workbooks whose predicted colour and
  tested door disagree.

- Crash avoidance, CBDA preprocess: the `Expected value` is the generic `TTC`
  for every dooring row, replacing the per-response `TTC @ t_door_opening` /
  `TTC @ t_warning` / `TTC @ t_information` text. Which TTC the lab reports
  now follows from the row's own `Type of system`.

- Crash protection, VRU legform / upper-legform selection: the L0/L1 (U0/U1)
  start-point draw is now unconditional, as VRU Impact v1.2 §3.4.3.3 /
  §3.4.4.3 describe it. Every all-blue legform row draws L0/U0 (even signed
  columns 0, ±2, ±4, …) or L1/U1 (odd signed columns ±1, ±3, …) and promotes
  the SA cells on the drawn columns to ST; T, ST and grey cells are untouched
  and every blue cell is still returned. Previously the draw only fired when
  an SA cell had no tested neighbour, and parity was measured on the position
  within the blue span rather than on the signed column. A no-orphan row that
  used to be emitted exactly as declared can now gain tested points (an aPLI
  Femur row with 8 declared points becomes 11 with L0 or 12 with L1, applied
  twice with Knee & Tibia). Because every preprocess now draws, a re-upload or score recompute
  re-rolls the selection; persisting the drawn parity is tracked separately.

### Removed

- Safe driving, Driver monitoring predictions: a
  (Task / glance target type, Movement type, Vehicle response) group may now
  hold both Red and Green cells. The `dm_group_consistency` and
  `dm_na_forced_red` consistency rules are gone, so such a sheet uploads and
  locks without a finding, and `preprocess` no longer overwrites the group's
  Green and `N/A` cells with `Red` in the preprocessed template -- they come
  back exactly as entered. `common.CHECK_DM_GROUP_CONSISTENCY` and
  `common.CHECK_DM_NA_FORCED_RED` are removed with the rules. Test-point
  selection and scoring are unchanged: the group is still suppressed in memory,
  so none of its gaze locations is selected, and it still scores 0 because the
  prediction cells are still not all Green. Cells that are Red for their own
  reason -- an explicit Red, or a required cell left blank -- are still written
  Red.

- Crash protection, VRU legform predictions: the T/ST/SA left/right symmetry
  check is gone, since Euro NCAP accepts asymmetric legform predictions.
  `preprocess` no longer rejects a T/ST or SA cell whose mirror cell has a
  different colour, and `find_consistency_findings` no longer emits
  `legform_symmetry` findings; `no_prediction_provided` and every other rule
  are unchanged. An ST point whose mirror has no measured value keeps the
  existing warning and scores 0. `common.CHECK_LEGFORM_SYMMETRY`,
  `vru_processing.check_legform_blue_symmetry`,
  `collect_legform_symmetry_violations` and `LegformSymmetryViolation` are
  removed with the check.

### Fixed

- Safe driving, ACC performance: `compute-score` on a preprocessed workbook
  read from disk now finds the `VA - ACC verif.` matrix columns
  (`OEM Prediction`, `VUT speed`, `Target speed`). They were read as unnamed
  columns, so every matrix point scored Red / 0. Scoring change for CLI
  users: ACC performance scores computed from a file go up to what the
  measured values earn. Callers that pass an in-memory DataFrame with named
  columns are unaffected. A sheet whose matrix columns still cannot be
  matched now fails with a `ValueError` instead of scoring 0.

- Safe driving, Occupant presence: on a vehicle without a child presence
  system, the Crash occupancy row of `OM - Occ. presence verif.` shows
  `General requirements` in its `Scenario` cell instead of a blank cell, in
  both the preprocessed workbook and the report. Scores are unchanged. The
  DataFrame returned by `occupant_presence.compute_classification_score` no
  longer blanks a `Score` equal to the row above (the CLI report does not
  write that column). `common.opposite_ffill` takes a new optional `group_column` argument;
  omitting it keeps the previous behaviour.

- Crash protection, Rear Impact: the Static-Rear seat geometry is scored as
  the protocol's points ladder instead of the lowest of the four static
  criteria. Per seat row: both effective heights met 0.375, then
  `Backset - Mid` +0.25, then `Backset - Lowest` (worst case) +0.125, then
  a non-use PASS +0.25; halved per row when there is a third row. A failed
  backset no longer zeroes the row nor withholds the non-use credit; only a
  failed effective height does. The Static-Rear `Body regionscore` now
  reads 0 / 50 / 83.33333 / 100. A criterion counts as met when it is not
  red, so an effective height of exactly 720 / 770 mm passes. Scores change
  for files with a red or blank backset.

- Safe driving, Speed assistance: a failed `SLIF - General requirements` now
  scores `SLIF - System updates` 0 as well. It was skipped, so the row stayed
  blank (Speed assistance unassessed) or kept points from an earlier
  calculation. Protocol-driven scoring fix: VA v1.2 §1 makes the SLIF points
  conditional on the General Requirements.

- Safe driving, Speed assistance: in the SLIF Advanced Speed Limits
  prediction grid, text other than `Green`/`Red` (e.g. a pasted `Yellow` or a
  typo) now fails the row instead of counting as blank / not applicable.
  Blank, `N/A` and `-` cells are unchanged. Scoring change for malformed input
  only; the template's Green/Red dropdown never produces it.

- Crash avoidance, CPLA / CBLA FCW rows: a warning at a TTC of exactly 1.70 s
  is now Green. FC v1.2 §5.2 gives Green for `TTC ≥ 1.7 s` and Red for
  `TTC < 1.7 s`; the band lookup gave the edge value to Red.

- Safe driving, ACC performance: the CCRb / CMRb green threshold is now the
  VUT speed minus 2 km/h (53 km/h on both template rows) instead of VUT minus
  target minus 2 (3 km/h, which every run cleared). The target brakes to a
  standstill and the VUT follows it, so the `Target speed` cell (50 km/h) is
  the target's initial speed, not its speed at Tend; "fully avoids" means the
  VUT is at a standstill within the 2 km/h tolerance. Protocol-driven scoring
  change (Safe Driving - Vehicle Assistance v1.2 §2.1.3 / §2.2.3): a Green
  prediction that previously scored full for any measured reduction above
  3 km/h now scores Orange (half) for reductions above 15 km/h up to and
  including 53 km/h, and Red (zero) for reductions of 15 km/h or less.
  Templates and the `VA - ACC pred.` cells are unchanged.

- Crash avoidance, verification-test selection: the cells preprocess keeps out
  of the lab's pool now match the v1.2 Corner Case clauses exactly. LSC
  `CMCscp SfS` 70/80 km/h and `CMFtap SfS` 80 km/h (LSC v1.2 §4.1.2, target
  speed >60 km/h) are no longer written to `LSC - Car & PTW verif.`; the
  evaluation tool draws them as Corner Cases from `CORNER_CASE_CELLS` instead,
  where it previously found nothing to draw. `CCRb` excludes only target speed
  >100 km/h (FC v1.2 §4.1.3), not >80, so the 90/100 km/h rows are back in
  the Extended-range pool. `CC ELK On/OvU/OvI` exclude target speed
  >100 km/h, not >80: LDC v1.2 §4.1.3 prints the overtaking clause for the CM
  ELK scenarios only, which Euro NCAP confirmed is a typo, and the CCRb
  threshold applies to Car-to-Car ELK as well. Only `CC ELK OvU` reaches above
  100 km/h in the template, so it joins `CORNER_CASE_CELLS` with its 24 cells
  (VUT 100/110/120/130 → target 110/120/130/140 × the six lateral velocities),
  all of them Extended-range; `CC ELK On` and `CC ELK OvI` stop at 100 km/h
  and stay out. `CORNER_CASE_CELLS` also
  drops the cells `CELL_REMOVAL_MAP` blanks from the matrix, which could never
  be drawn: the `CCFtap`/`CMFtap` 70 km/h column and the `CCCscp`/`CMCscp`
  70/80 km/h cells above VUT 30 km/h. Protocol-driven: the LSC
  verification sheet loses those rows, `LDC - Car & PTW verif.` loses its
  CC ELK OvU rows above 100 km/h, and the CCRb / CC ELK random selection
  changes. Already-issued verification sheets are not rewritten.

- Crash avoidance, prediction consistency check: the `FC - Driver State Link`
  grid is now required under the two Frontal Collisions elements, split by
  scenario row (CCRs…CMCscp → Car & PTW, CPLA…CBFA → Pedestrian & cyclist),
  instead of under Lane Departure Collisions / Single vehicle. A per-element
  LDC or LSC prediction upload no longer reports 45 `FC - Driver State Link …
  is missing` findings, and the FC uploads that actually fill the grid are now
  checked for it. The grid is the FC v1.2 §2.4.1 / §5.4 per-scenario claim;
  the LDC 3-point `Driver state link` loadcase is unrelated and its scoring
  is unchanged.

- Crash avoidance, `Initial position offset` robustness layer: the verification
  condition for `CC ELK On` / `CC ELK Ov` was `0.5m closer to VUT`; the protocol
  (CA 002 v1.2 §2.1.1 / §2.1.3, LDC v1.2 §5.2.4.1) specifies ±0.25 m on both
  sides, so it is now `0.25m closer to VUT`. Present since 5.4.5-alpha.10.
  Already-issued verification sheets are not rewritten by this change.

- Crash avoidance, `Initial position offset` robustness layer: the verification
  condition for `CM ELK On` / `CM ELK Ov` was the literal token
  `safety_margin_violation`, which no protocol defines. The layer is a ±0.25 m
  path offset for Car and PTW alike (LDC v1.2 §5.2.4.1 and Appendix B, CA 002
  v1.2 §2.1.2 / §2.1.4), so both now draw `0.25m away from VUT` /
  `0.25m closer to VUT` exactly like `CC ELK On` / `CC ELK Ov`. Present since
  5.4.5-alpha.10. Already-issued verification sheets keep the token and are
  not rewritten; the change is score-neutral for them.

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
