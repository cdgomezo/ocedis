# Code Architecture

## Layout

```
src/ocean14c/
├── glodap.py             # load GLODAP 1995 field, define surface slice
├── regression.py         # two-population dDelta14C/dt regression (hardcoded coeffs)
├── regions.py             # Southern Ocean / N.Pacific / Drake Passage masks
├── surface_ocean.py       # project Delta_oce(t) forward from 1995
├── atm_stations.py        # parse raw station Delta14C records (2 formats)
├── atmosphere.py           # curve-fit, splice, tropical band -> Delta_atm(lat,t)
├── atmosphere_forcing.py  # download + process ERA5 winds -> <U^2>
├── winds.py                # load ocean14c.atmosphere_forcing's output
├── gas_exchange.py         # Schmidt number, transfer velocity k
├── solubility.py            # Weiss (1974) CO2 solubility
├── takahashi.py             # pCO2 climatology, regrid, drift
├── foa.py                   # combine k, s, pCO2 -> Foa, with unit conversions
├── isoflux.py                # Foa * Delta_odis, aggregation, TM5 umol m-2 s-1 conversion
├── output.py                 # NetCDF writer: flux_c14.<term>.<year>.nc
├── table1.py                  # Table 1-matched flux variant (per-year rescaling)
├── pipeline.py                 # ties every module together: reconstruct_year(year)
└── validate.py                # Table 1 / Table S2 targets, comparison, gamma fit

scripts/
├── validate_full_series.py   # loops reconstruct_year() over a year range,
│                              # writes results/full_series_results.csv and
│                              # results/figures/{temporal,spatial}_comparison.png,
│                              # plus the flux product(s) chosen by --flux-variant
└── plot_flux_variants.py     # comparison/verification figures for the two
                               # flux products, read back from the shipped
                               # NetCDF files (results/figures/flux_variants_*)
```

See [`docs/SCIENCE.md`](SCIENCE.md) §3 for the data-flow diagram connecting
these, and each module's own docstring for the specific CLAUDE.md section
and equation it implements — every module opens with that cross-reference.

## Conventions

**Grids are `xarray.DataArray`s**, dims named `lat`, `lon` (and `depth`,
`month` where relevant). Longitude is always standardized to **−180..180**
(GLODAP's native convention); `winds.py` converts ERA5's native 0–360°
convention on load. This means any two fields from different loaders can be
combined (`+`, `*`, `.sel(...)`) without manual alignment, as long as their
underlying 1°×1° grids agree — xarray handles the coordinate matching.

**Units are explicit in function signatures and docstrings**, not implicit.
`foa.py` in particular documents every conversion constant
(`CM_PER_HR_TO_M_PER_YR`, `L_TO_M3`, `UATM_TO_ATM`) rather than folding them
into a single opaque prefactor — this was a deliberate choice so unit bugs
are visible and testable in isolation (`test_foa.py` pins the composed
conversion factor down directly).

**Config toggles live in `config/default.yaml`**, not scattered magic
numbers in code — see [`docs/USAGE.md`](USAGE.md) for what each one does.
Every toggle corresponds to a genuine literature ambiguity documented in
`docs/SCIENCE.md`, not an implementation shortcut.

**Real-data tests skip gracefully.** Any test that needs
`data/raw/<something>/` is guarded with
`pytest.mark.skipif(not path.exists(), ...)` (see e.g. `test_glodap_grid.py`,
`test_atm_stations.py`) — the full suite is green with or without the raw
data present, but running with it present is what actually exercises the
format-parsing logic end-to-end.

## Module reference

| Module | Purpose | Key functions |
|---|---|---|
| `glodap.py` | Load & parse the GLODAP 1995 field | `load_c14_grid`, `surface_delta_1995`, `load_stations` |
| `regression.py` | Two-population rate-of-change regression | `rate_of_change`, `COEFFICIENTS` |
| `regions.py` | Spatial masks for the regression populations | `southern_ocean_mask`, `northern_north_pacific_mask`, `drake_passage_mask`, `is_cold_population` |
| `surface_ocean.py` | Project Δoce forward from 1995 | `project_delta_oce` |
| `atm_stations.py` | Parse raw Δ¹⁴C station files | `load_c14_station`, `load_drp_curl_event` |
| `atmosphere.py` | Δatm(lat, t) reconstruction | `fit_year_edge_values`, `splice_station_series`, `delta_atm_field` |
| `atmosphere_forcing.py` | Download/process ERA5 winds | `process_year`, `download_all`, CLI `main` |
| `winds.py` | Load processed ERA5 `<U²>` | `load_u2` |
| `gas_exchange.py` | Schmidt number & transfer velocity | `schmidt_number`, `transfer_velocity`, `DEFAULT_GAMMA` |
| `solubility.py` | CO2 solubility | `co2_solubility` |
| `takahashi.py` | pCO2/SST climatology | `load_pco2_climatology`, `load_sst_climatology`, `regrid_nearest`, `project_pco2_ocean` |
| `foa.py` | One-way ocean→atmosphere flux | `compute_foa`, `foa_flux`, `foa_flux_to_pgc_per_yr` |
| `isoflux.py` | Combine & aggregate | `compute_isoflux`, `spatial_sum`, `flux_weighted_mean`, `to_umol_flux` |
| `output.py` | TM5 flux product writer | `write_year(..., term=)` -> `flux_c14.<term>.<year>.nc`, variable `c14flux` [umol m-2 s-1] |
| `table1.py` | Table 1-matched flux variant | `scale_factors`, `scale_flux`, `TERM_RECONSTRUCTION`/`TERM_TABLE1` (docs/SCIENCE.md 8.3) |
| `pipeline.py` | End-to-end orchestration for one year | `reconstruct_year`, `cell_area_m2` |
| `validate.py` | Targets & comparison | `TABLE_1_FOA_DELTA_ODIS`, `TABLE_S2`, `compare_series`, `fit_gamma`, `load_ocediseq_annual`, `spatial_correlation` |

## Validation status

`ocean14c.pipeline.reconstruct_year()` wires every module together for one
year; `scripts/validate_full_series.py` loops it over a year range and
produces the comparison table/figures. It has been run for real for the
**full 2000–2012 series** (real GLODAP, Takahashi, Delta_atm, and real
downloaded-and-processed ERA5 winds for all 13 years) and checked against
Table 1 (including the `SUM` row), Table S2, and the paper's own gridded
product (`Ocediseq.nc`) — see
[`docs/SCIENCE.md`](SCIENCE.md#8-validation-results) for the actual
numbers and `results/figures/` for the plots. Headline results: **0.91–0.93
spatial Pearson correlation** against `Ocediseq.nc`, every year
(scale-invariant, so independent of the γ calibration choice); **SUM
2000–2012 within −11.8%** of Table 1's published −29,683 PgC‰ with a γ
refit against Table S2's 13-year Global_Foa series, stable across the whole
period rather than concentrated in one or two years.

That run also caught a real bug the way integration tests are supposed to:
`takahashi.regrid_nearest()`'s naive nearest-neighbor interpolation was
propagating `NaN` from Takahashi's coarse native land mask onto genuinely
oceanic 1°×1° target cells, silently dropping **62% of GLODAP's ocean-cell
area** from the Foa calculation before the fix (see the git history / test
`test_regrid_nearest_skips_nan_native_cells` for the fix, which searches for
the nearest *valid* native cell instead of the nearest cell period). This is
exactly the kind of bug that's invisible to per-module unit tests with
small synthetic fixtures and only shows up once real data flows through the
whole pipeline — worth remembering next time something "unit-tests green"
but the aggregate numbers look off by a suspiciously round factor.

## Test suite

```
tests/
├── test_glodap_grid.py            # real-data: shape, fill value, row-order regression check
├── test_regression_coefficients.py
├── test_regions.py
├── test_surface_ocean.py
├── test_atm_stations.py           # real-data: parses all 6 stations
├── test_atmosphere.py
├── test_atmosphere_forcing.py     # mocked cdsapi calls, synthetic NetCDF fixtures
├── test_winds.py
├── test_schmidt_number.py
├── test_solubility.py
├── test_takahashi.py              # real-data: known value, dims
├── test_foa.py
├── test_isoflux.py
├── test_output.py
└── test_validate.py
```

Built test-first throughout (see [`docs/DEVELOPMENT.md`](DEVELOPMENT.md) for
the workflow) — every module has at least one test anchored to a real,
citable number from the primary literature where one exists (Sc(20°C)=668,
K0(20°C,S35)=0.0324 mol/L·atm, the regression coefficients, Table 1/S2), not
just internal-consistency checks.
