# Development

## Workflow this codebase was built with

Every module in `src/ocean14c/` was built test-first: a failing test
written and run (confirming it fails for the right reason — missing
function/module, not a typo), then the minimal implementation to pass it,
then re-run to confirm green, before moving to the next behavior. Several
bugs were caught this way that wouldn't have shown up in a quick manual
check — e.g. `regions.py`'s Southern Ocean flood-fill initially broke on
real data because the South Pole grid cells are `NaN` (no bottle samples
under permanent ice), which a synthetic test fixture without `NaN` gaps
didn't exercise; the fix (treat missing data as "still connected," not "not
connected") came from re-running against the real GLODAP field, not just
the unit tests.

**If you're extending this code, keep doing that**: write the test, watch
it fail, implement, watch it pass. It's especially valuable here because so
much of this pipeline is unit conversions and literature-sourced constants
— exactly the kind of code where a silent transposition or sign error
produces a plausible-looking wrong number instead of a crash.

## Extending the pipeline

`ocean14c.pipeline.reconstruct_year()` is the end-to-end orchestration, and
`scripts/validate_full_series.py` is the multi-year runner — it loops
`reconstruct_year()` over a year range, compares against `TABLE_1_SUM` /
`TABLE_S2` / `Ocediseq.nc`, and writes the temporal and spatial comparison
figures under `results/figures/`. Both the full 2000–2012 series and every
year's ERA5 data are already fetched and validated — see
[`docs/ARCHITECTURE.md`](ARCHITECTURE.md#validation-status) and
[`docs/SCIENCE.md`](SCIENCE.md#8-validation-results) for the current
numbers (SUM within −11.8% of Table 1 with a refit γ, spatial correlation
0.91–0.93 every year). What would close the remaining gap further:

1. **Look at the Southern Ocean boundary specifically.** The difference
   maps in `results/figures/spatial_comparison.png` show the disagreement
   isn't spatially random — it concentrates in a narrow band right at the
   Southern Ocean's poleward edge, consistently across all plotted years
   (the reconstruction runs less negative than `Ocediseq.nc` there). The
   likely culprits are `regions.southern_ocean_mask()`'s dynamic 20‰-isoline
   boundary or Drake Passage handling differing slightly from the paper's
   own implementation — this is a much more targeted lead than "the
   reconstruction is off by ~12%, go find out why."
2. **Consider tightening the atmospheric curve fit.** The current
   `atmosphere.py` implementation (polynomial + harmonics via least squares)
   is a simplified stand-in for NOAA's `ccgcrv` (see
   [`docs/SCIENCE.md`](SCIENCE.md#51-station-curve-fitting)) — reproducing
   that more faithfully (particularly its low-pass-filtered residual) would
   likely tighten the Δodis-driven share of the remaining gap.
3. **Rerun `fit_gamma()`** if you change anything upstream of `Foa` (SST
   loading, Schmidt polynomial, wind regridding) — the current γ=0.2147 is
   fit to the *current* pipeline's other approximations, not a universal
   constant independent of them.
4. A natural home for further iteration is
   `notebooks/01_reconstruct_ocean_isoflux.ipynb` (the directory already
   exists, per the structure `CLAUDE.md` §8 originally sketched) — useful
   for interactively poking at the Southern Ocean boundary question above,
   which is easier to debug with quick re-plots than by editing and
   re-running the script repeatedly.

Beyond that, `CLAUDE.md` §10 sketches extending to the full atmospheric ¹⁴C
mass balance (fossil, terrestrial disequilibrium, cosmogenic/nuclear
production, the two-box troposphere/stratosphere transport model) — this
repo's ocean term is one of five in that eventual full budget.

## Adding a new module

Follow the pattern every existing module uses:

1. Docstring at the top citing the specific `CLAUDE.md` section (or
   `docs/SCIENCE.md` section) and equation being implemented.
2. `xarray.DataArray` in/out for anything gridded, dims named `lat`/`lon`
   (see [`docs/ARCHITECTURE.md`](ARCHITECTURE.md#conventions)).
3. Explicit units in docstrings and, for anything involving a unit
   conversion, named constants (see `foa.py`) rather than folded-in magic
   numbers.
4. A test file with at least one assertion anchored to a real, citable
   number from the literature, not just internal self-consistency, where
   the module's subject has one available.
5. If the module parses a new raw data format, add a section to
   [`docs/DATA.md`](DATA.md) documenting the format and any
   empirically-resolved ambiguities (the GLODAP row-ordering writeup there
   is the template for how much detail is worth recording).

## Config toggles

`config/default.yaml` documents three literature ambiguities (regression
coefficient version, tropical Δatm band, gas-transfer γ / Schmidt
polynomial) as YAML comments. It is **not currently loaded by any code** —
each toggle maps to a keyword argument on the relevant function (see
`docs/USAGE.md`'s "Adjusting configuration" section for the exact mapping).
Wiring an actual config loader (e.g. reading the YAML and threading values
through a pipeline run) is reasonable follow-up work once the orchestration
script above exists — don't add one speculatively before there's a caller
that needs it.

## Repository status

- **No CI configured.** Run `pytest tests/` locally before committing.
- **No CI workflow.** Worth adding one that runs `pytest tests/`: the
  real-data tests skip when the raw data is absent, which given its licence
  and size is what CI should do, so a workflow would exercise the
  synthetic-fixture tests.
- **No LICENSE file yet.** Add one before making the repository public —
  this affects how the raw third-party data licensing questions in
  `docs/DATA.md` interact with redistribution, so it's worth a deliberate
  choice rather than a default.

## Dependency updates

Dependencies are declared in `pyproject.toml`. Core dependencies are
intentionally minimal (numpy/scipy/xarray/pandas/matplotlib/netCDF4/pyyaml);
`cdsapi` is isolated in the `era5` extra since it's only needed for the wind
download, not for running the rest of the pipeline against already-fetched
data. After changing `pyproject.toml`, reinstall with
`pip install -e ".[dev]"` (or `".[dev,era5]"`) and re-run the test suite.
