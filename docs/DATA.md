# Data

All raw, third-party data lives under `data/raw/` and is **gitignored** —
nothing in this directory is checked in (large binary files, mixed
licensing). This document is the map you need to reconstruct it, plus the
exact on-disk formats each loader expects (several of which took real
reverse-engineering to pin down and are recorded here so nobody has to
redo that work).

```
data/
├── raw/
│   ├── GLODAP/            # Key et al. (2004) gridded surface delta14C-DIC
│   ├── Takahashi2009/     # Takahashi et al. (2009) pCO2 climatology
│   └── Delta_atm/         # the raw atmospheric delta14C station records
├── processed/
│   ├── era5_u2/           # output of ocean14c.atmosphere_forcing, u2_<year>.nc
│   └── era5_hourly/       # output of data/ERA5_hourly_regional.py
└── ERA5_hourly_regional.py    # regional hourly wind and SST fetcher
```

Three additional reference NetCDF files sit at the **repo root** (also
gitignored — see below for why): `Ocediseq.nc`, `Ocean_d14C_2004.nc`,
`Ocean_d14C_trend.nc`.

## GLODAP (`data/raw/GLODAP/`)

**Source:** Key et al. (2004) gridded surface Δ¹⁴C-DIC climatology
(https://glodap.info — the CDIAC link cited in the original paper is dead).

**Files:**
- `C14.data.txt`, `C14.err.txt` — the gridded field and its uncertainty.
- `Lat.centers.txt` (180 values, −89.5 to 89.5), `Long.centers.txt` (360
  values, −179.5 to 179.5), `Depth.centers.txt` / `DepthEdges.txt` (33
  depth levels, 0–5500 m).
- `C14Sta.0.txt` — 936 raw bottle-sample stations (`longitude, latitude,
  c14`). Not needed for the gridded product itself, but essential for
  validating the loader (see below).

**Row-major ordering (empirically resolved, not documented anywhere in the
files themselves):** `C14.data.txt` has 11,880 rows = 360 lon × 33 depth,
each row holding 180 comma-separated latitude values. The row index is
**`depth_idx * n_lon + lon_idx`** (depth-major, longitude fastest) — *not*
the other plausible ordering (`lon_idx * n_depth + depth_idx`). This was
determined by comparing gridded surface values against the raw
`C14Sta.0.txt` stations at their nearest grid cell: the correct ordering
gives a median absolute error of ~11‰ against individual bottle samples,
the wrong ordering ~64‰ (and drops ~300 stations onto missing-data cells
that the correct ordering doesn't). `test_glodap_grid.py` locks this in as
a regression test.

**Fill value:** `-999` means missing (land or no data); `ocean14c.glodap`
converts this to `NaN`.

**"Surface"**: Miller et al. (2025) define surface as <50 m —
`ocean14c.glodap.surface_delta_1995()` averages depths 0/10/20/30 m.

## Takahashi 2009 (`data/raw/Takahashi2009/`)

**Source:** Takahashi et al. (2009) pCO2 climatology, usually distributed
via NOAA/OCADS.

**Files:**
- `month_flux_2006c.txt` — monthly, 4°×5° grid (lat centers every 4°
  starting at 0, lon centers every 5° starting at 2.5°E, native 0–360°E
  convention). Header/column-name row is at **line 109** (1-indexed);
  `ocean14c.takahashi.load_pco2_climatology()` skips to it with
  `pd.read_csv(..., skiprows=108)`. Column of interest: `PCO2_SW`, already
  normalized to a **year-2000 reference at +1.5 μatm/yr** — the same drift
  rate CLAUDE.md and the Miller 2025 text specify, confirmed independently
  from this file's own header documentation.
- `sumflux_2006c.txt` — annual summary, same grid. Not currently consumed
  by any loader; kept for reference/cross-checking.

**Regridding:** native 4°×5° → target 1°×1° via nearest-neighbor
(`ocean14c.takahashi.regrid_nearest()`).

## Atmospheric Δ¹⁴C stations (`data/raw/Delta_atm/`)

Six real station records, covering all six stations CLAUDE.md §3.4 names.
**Two distinct formats** are present:

### `.c14` format (NWR, JFJ, SPO, PSA, SMO — and a partial DRP file)

Semicolon-delimited, `#`-commented metadata header, then a header row
starting with `#Site;...`:

```
Site;SamplingHeight;Year;Month;Day;Hour;Minute;DecimalDate;IntegrationTime;
SamplingPattern;14C;WeightedStdErr;NbPoints;Flag;...
```

`ocean14c.atm_stations.load_c14_station()` keeps only rows with
**`Flag == 'O'`** ("correct after quality control" — see the file's own
`# FLAG:` comment block for the full U/N/O/K legend).

Coverage windows in the actual files (comfortably bracket what's needed):
NWR 2003–2021, JFJ 1986–2024, SPO 1999–2024, PSA 2005–2007, SMO 2001–2023.

### ObsPack-style format (DRP only)

`co2c14_drp_shipboard-flask_376_curl_event.txt` — space-delimited, NOAA
ObsPack convention, header row:

```
site_code year month day hour minute second datetime time_decimal
air_sample_container_id value value_unc latitude longitude altitude
elevation intake_height method event_number instrument analysis_datetime qcflag
```

**Why DRP needs this second file:** DRP's own `.c14` file only covers
2009–2011, missing the 2006–2008 window Miller et al. (2025) use. This
ObsPack file covers 2005–2017, so `ocean14c.atm_stations.load_drp_curl_event()`
is used for DRP instead of the `.c14` parser.

**QC filter (documented in the file's own header, lines ~167–169):** the
3-character `qcflag` has column 1 = REJECTION flag, column 2 = SELECTION
flag, column 3 = INFORMATION flag. Only non-`.` characters in columns 2/3
are informational (e.g. preliminary data) and don't invalidate the
measurement. **Only column 1 must be `.`** for a row to be usable — this is
what `load_drp_curl_event()` checks (`row[flag_col][0] == "."`), *not* a
naive "flag string equals `...`" check, which would have discarded a
third of otherwise-valid rows.

## ERA5 winds

**Not raw data you download and place in a directory** — instead, run
`python -m ocean14c.atmosphere_forcing` (needs a personal CDS API token in
`~/.cdsapirc`; see [`docs/USAGE.md`](USAGE.md)). It downloads 6-hourly
10 m u/v wind components, computes monthly `<U²>`, and writes
`data/processed/era5_u2/u2_<year>.nc` — one small file per year, deleting
the ~300MB raw yearly download immediately after processing.

`data/ERA5_hourly_regional.py` is its regional, hourly counterpart, writing
`data/processed/era5_hourly/` one month at a time. The hourly flux needs
hourly wind, because the transfer velocity goes as U² and a monthly mean
cannot be unpicked into one.

## Ground-truth reference files (repo root)

- **`Ocediseq.nc`** (180×360×156, i.e. 1°×1°, 156 months = exactly
  2000–2012): the paper's own published gridded ocean-disequilibrium isoflux
  dataset (doi:10.15138/8dhy-6k90) — the actual answer key.
- **`Ocean_d14C_2004.nc`**, **`Ocean_d14C_trend.nc`**: reference fields on a
  coarser (73×97) model grid, of unconfirmed provenance.

**These are used only for validation/cross-check, never as pipeline
inputs** — the whole point of this codebase is an independent
reconstruction from raw data that can be checked against them, not a
pass-through of the answer key. They're large binary files and gitignored;
if you don't have them, everything except the final ground-truth comparison
in `validate.py` still works.

## Reference literature (PDFs, repo root)

All gitignored (large, copyrighted). See [`CLAUDE.md`](../CLAUDE.md) §2 for
the complete annotated reading list and priority order, and
[`docs/SCIENCE.md`](SCIENCE.md) §8 for the bibliography of what's actually
load-bearing for the code.
