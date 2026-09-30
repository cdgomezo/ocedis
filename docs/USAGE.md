# Usage

## Install

```bash
python3 -m pip install -e ".[dev]"
```

Requires Python ≥3.10. Core dependencies: `numpy`, `scipy`, `xarray`,
`pandas`, `matplotlib`, `netCDF4`, `pyyaml`. `pytest` comes with the `dev`
extra. The ERA5 download additionally needs the `era5` extra (see below).

This was developed and is tested against the environment's `python3`
interpreter with those packages available — if you're managing your own
virtual environment, make sure it has them (`pip install -e .` above will
pull them in from `pyproject.toml`).

## Running the tests

```bash
pytest tests/
```

Tests that need real downloaded data (GLODAP, Takahashi, the raw Δ¹⁴C
station records — see [`docs/DATA.md`](DATA.md)) skip automatically if
`data/raw/...` isn't present, so the suite is green either way. To exercise
the full suite including real-data checks, populate `data/raw/` per
`docs/DATA.md` first.

## Downloading ERA5 winds

```bash
python3 -m pip install -e ".[era5]"
```

You need a personal CDS API token in `~/.cdsapirc` (get one at
https://cds.climate.copernicus.eu/). Then:

```bash
python3 -m ocean14c.atmosphere_forcing
```

This downloads 2000–2012 (matching Miller et al. 2025's analysis period),
one year at a time, computing monthly `<U²>` and writing
`data/processed/era5_u2/u2_<year>.nc`, deleting each year's ~300MB raw
download as soon as it's processed. **It's resumable** — if interrupted,
re-running skips any year whose output file already exists. This is a
long-running operation against your personal CDS quota; there's no need to
run it more than once.

Options:

```bash
python3 -m ocean14c.atmosphere_forcing --output-dir data/processed/era5_u2 --start-year 2000 --end-year 2012
```

## Example: reconstructing Δodis for one year

This is real, runnable code (verified against the actual data in this
repo) showing how the modules compose. It only needs GLODAP and the
atmospheric station data — not ERA5/Takahashi, which are only needed for
the `Foa` half of the pipeline (see below).

```python
from ocean14c.glodap import surface_delta_1995
from ocean14c.surface_ocean import project_delta_oce
from ocean14c.atm_stations import load_c14_station, load_drp_curl_event
from ocean14c.atmosphere import (
    splice_station_series, delta_atm_field,
    NORTHERN_HEMISPHERE_WINDOWS, SOUTHERN_HEMISPHERE_WINDOWS,
)

YEAR = 2006

# 1. Surface ocean Delta14C, projected forward from the 1995 GLODAP field
delta_1995 = surface_delta_1995("data/raw/GLODAP")
delta_oce = project_delta_oce(delta_1995, year=YEAR)

# 2. Atmospheric Delta14C for YEAR, from the raw station records
d = "data/raw/Delta_atm/"
nh_series = {
    "NWR": load_c14_station(d + "noaa_l2_2025_1_nwr_3m_spot_min_clean.c14"),
    "JFJ": load_c14_station(d + "uheicrl_l2_2025_1_jfj_5m_int_14day_clean.c14"),
}
sh_series = {
    "SPO": load_c14_station(d + "ucsd_l2_2025_1_spo_2m_spot_1min_clean.c14"),
    "PSA": load_c14_station(d + "ucsd_l2_2025_1_psa_10m_spot_min_clean.c14"),
    "SMO": load_c14_station(d + "ucsd_l2_2025_1_smo_2m_spot_1min_clean.c14"),
    "DRP": load_drp_curl_event(d + "co2c14_drp_shipboard-flask_376_curl_event.txt"),
}
nh = splice_station_series(nh_series, NORTHERN_HEMISPHERE_WINDOWS, [YEAR])
sh = splice_station_series(sh_series, SOUTHERN_HEMISPHERE_WINDOWS, [YEAR])
delta_atm = delta_atm_field(delta_oce["lat"], delta_n=nh[YEAR], delta_s=sh[YEAR])

# 3. Delta_odis = Delta_oce - Delta_atm
delta_odis = delta_oce - delta_atm
print("global mean Delta_odis:", float(delta_odis.mean(skipna=True)))
```

For 2006 this prints a global mean Δodis around **−36‰** — in the right
ballpark next to Table S2's flux-weighted 2006 value of −32.9‰ (it won't
match exactly: this is an unweighted spatial mean, not the flux-weighted
mean Table S2 reports — see `docs/SCIENCE.md` §7 for why those two numbers
are legitimately different quantities).

### The full pipeline, end to end

`ocean14c.pipeline.reconstruct_year()` does everything shown above (and the
`Foa` half — SST, pCO₂, ERA5 winds, gas exchange, solubility) in one call.
It needs the ERA5 download for that year done first (see above):

```python
from ocean14c.pipeline import reconstruct_year

result = reconstruct_year(2006)
print(result["global_isoflux"])       # PgC permil/yr, straight sum -- compare to Table 1
print(result["global_foa"])           # PgC/yr -- compare to Table S2
print(result["global_delta_odis"])    # permil, flux-weighted -- compare to Table S2
print(result["socean_foa"], result["socean_delta_odis"])
```

This has been run for real against the full 2000–2012 series: **0.91–0.93
spatial correlation** against the paper's own gridded product
(`Ocediseq.nc`) every year, and the 13-year **SUM within −11.8%** of Table
1's published value with a refit γ. Full numbers and discussion in
[`docs/SCIENCE.md`](SCIENCE.md#8-validation-results).

## Running the full multi-year validation + plots

The easiest way to reproduce the full comparison (all 13 years, both the
temporal and spatial figures) is the script that already did it:

```bash
python3 scripts/validate_full_series.py --gamma 0.21469736017350816 --map-years 2000,2006,2012
```

Needs every year's ERA5 data already processed (`python -m
ocean14c.atmosphere_forcing`, ~13 CDS requests — the script skips years
whose ERA5 file is missing rather than failing, so you can run it on a
partial download). Writes `results/full_series_results.csv` and
`results/figures/{temporal,spatial}_comparison.png`, plus the TM5 flux
product `output/flux_c14.ocedis.<year>.nc` (variable `c14flux`,
µmol C_D14C m⁻² s⁻¹ — see the README's "TM5 flux product" section). Pass
`--flux-dir none` to skip writing those, or `--flux-dir <path>` to redirect
them. `--flux-variant table1` (or `both`) additionally writes
`flux_c14.ocedis_table1.<year>.nc`, the same field rescaled per year onto
Table 1's published annual values — see the note below. Every year is cross-checked by converting `c14flux` back to PgC‰/yr
and comparing against the validated `global_isoflux`; the script aborts on
any mismatch. Omit `--gamma` to use
the default (γ=0.31, Wanninkhof 1992) instead of the value refit against
Table S2 — see [`docs/SCIENCE.md`](SCIENCE.md#8-validation-results) for how
much that changes the result. `--map-years` controls which years get full
spatial maps (all years are always included in the temporal comparison and
the CSV).

### Producing the Table 1-matched flux variant

```bash
python3 scripts/validate_full_series.py --gamma 0.21469736017350816 \
    --start-year 2000 --end-year 2025 --curve-fit ccgcrv --flux-variant both
```

(`--curve-fit ccgcrv` because it is what the shipped `output/` product was
built with, and because 2024–2025 extrapolate past the end of the
southern-hemisphere records and need it — §9. The whole 26-year run takes
about a minute.)

Writes both products into `--flux-dir`:

| file | what it is |
|---|---|
| `flux_c14.ocedis.<year>.nc` | the reconstruction — unchanged, still the default |
| `flux_c14.ocedis_table1.<year>.nc` | the same field × one scalar per year, so its annual global isoflux equals Miller et al. (2025) Table 1 exactly |

The scale factors are computed live from this run's own `global_isoflux`, so
they follow `--gamma`, `--regression-version` and `--curve-fit` automatically;
they are printed per year and recorded both in the CSV (`table1_scale`) and in
each file's `scaling` global attribute. Years after 2012, which Table 1 does
not cover, take the mean of the in-window factors (≈1.13); if the run contains
no year Table 1 covers, the script says so loudly and falls back to a frozen
constant for that curve fit. Everything else the script prints — the validation numbers,
the CSV, the figures — always describes the *reconstruction*, because that is
the thing being validated.

Downstream, the two are distinguished by the `<term>` in the filename,
`ocedis` against `ocedis_table1`, so neither can overwrite the other and a
consumer selects one by name.

Read [`docs/SCIENCE.md` §8.3](SCIENCE.md#83-the-table-1-matched-flux-variant)
before using the `table1` variant for anything you intend to publish — it
enforces agreement rather than earning it.

### Comparing and verifying the two products

```bash
python3 scripts/plot_flux_variants.py [--flux-dir output] [--year 2011]
```

Reads the **shipped NetCDF files** rather than re-running the pipeline — these
are verification figures, so they have to describe the bytes TM5 will be given.
Takes ~5 s and writes `results/flux_variants_comparison.csv` plus three figures
into `results/figures/`:

| figure | the question it answers |
|---|---|
| `flux_variants_temporal.png` | do the two products differ the way they should across 2000–2025, does the variant land on Table 1 (residual panel, in ppm against float32's own precision), and what is the factor doing year by year? |
| `flux_variants_spatial.png` | is the difference uniform, or did the rescaling deform the field somewhere? Four maps: both products on a shared scale, their difference, and their ratio at ±1 ppm. |
| `flux_variants_pointwise.png` | the same question where a map cannot resolve it — seasonal cycle, every cell × every month against the fitted line, and the spread of the pointwise ratio. |

It also prints the table it plots, so the numbers are readable without opening
a PNG.

## Validating against Table 1 / Table S2 / Ocediseq.nc manually

```python
from ocean14c.pipeline import reconstruct_year
from ocean14c.validate import (
    TABLE_1_FOA_DELTA_ODIS, TABLE_S2, compare_series,
    fit_gamma, load_ocediseq_annual, spatial_correlation,
)

# single year against Table 1 / Table S2:
result = reconstruct_year(2006)
print(compare_series({2006: result["global_isoflux"]}, TABLE_1_FOA_DELTA_ODIS))

# refit gamma against Table S2's Global_Foa for this year:
def foa_at_gamma(gamma):
    return {2006: reconstruct_year(2006, gamma=gamma)["global_foa"]}
fitted_gamma = fit_gamma(foa_at_gamma, target={2006: TABLE_S2[2006]["global_foa"]})

# spatial pattern against the paper's own gridded product:
ours = result["annual_isoflux"].transpose("lat", "lon")
theirs = load_ocediseq_annual("Ocediseq.nc", 2006).transpose("lat", "lon")
print(spatial_correlation(ours, theirs))  # 0.916 for 2006

# loop over multiple years (each needs that year's ERA5 data downloaded first):
computed = {year: reconstruct_year(year)["global_isoflux"] for year in range(2000, 2013)}
print(compare_series(computed, TABLE_1_FOA_DELTA_ODIS))
```

### Cross-checking the atmospheric reconstruction independently

This doesn't need ERA5 either, and is a good sanity check to run after
touching `atmosphere.py` or `atm_stations.py`:

```python
from ocean14c.atmosphere import splice_station_series, NORTHERN_HEMISPHERE_WINDOWS, SOUTHERN_HEMISPHERE_WINDOWS
from ocean14c.atm_stations import load_c14_station, load_drp_curl_event
# ... load nh_series / sh_series as above ...
years = list(range(2000, 2013))
nh = splice_station_series(nh_series, NORTHERN_HEMISPHERE_WINDOWS, years)
sh = splice_station_series(sh_series, SOUTHERN_HEMISPHERE_WINDOWS, years)
# compare nh[year]/sh[year] against Table 1's published Delta14CN/Delta14CS
# (hardcode from CLAUDE.md Sec 6, or from Table 1 in the Miller 2025 PDF directly)
```

Expect agreement to within roughly 0.3–3‰/year across 2000–2012 (see
`docs/SCIENCE.md` §5.1) — a much larger discrepancy than that signals a real
regression, not curve-fit noise.

## Adjusting configuration

Edit `config/default.yaml` (not yet wired into the modules as a loaded
config object — currently each toggle corresponds to a keyword argument on
the relevant function; see the file's comments for which argument maps to
which toggle, e.g. `regression.coefficients_version` → `version=` on
`rate_of_change()` / `project_delta_oce()`, `gas_exchange.gamma` → `gamma=`
on `transfer_velocity()` / `compute_foa()`).
