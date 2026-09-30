# ocean14c

Ocean ¹⁴C isotopic disequilibrium isoflux, reconstructed from first principles.

The ocean is a ¹⁴C source or sink to the atmosphere not because carbon moves
in one direction, but because the carbon going each way carries a different
Δ¹⁴C. This package reconstructs that term:

```
Δodis(lat, lon, t) = Δoce(lat, lon, t) − Δatm(lat, t)     disequilibrium [permil]
Foa(lat, lon, t)   = k · s · pCO2_ocean                   one-way ocean-to-air CO2 flux
isoflux            = Foa × Δodis                          [PgC permil yr-1]
```

Every input is rebuilt independently: the surface ocean Δ¹⁴C from a 1995
climatology projected forward by an empirical two-population regression, the
atmospheric Δ¹⁴C from a 22-station network, the gas transfer velocity from
reanalysis winds, and the pCO₂ climatology with its secular drift. The
published gridded product of Miller et al. (2025) is used only as a validation
target, never as an input.

It reproduces that paper's spatial pattern to a Pearson r of 0.91 to 0.93 in
every year of 2000–2012, and its 13-year total to within 12 %, inside the
±20 % structural uncertainty the gas transfer velocity alone carries.

## Products

**The diagnostics the literature reports**, as annual global and Southern
Ocean totals in PgC ‰ yr⁻¹, with the Southern Ocean defined dynamically by the
20 ‰ isoline rather than a fixed latitude.

**A gridded flux**, monthly on a 1°×1° global ocean grid, as the composite
tracer C_D14C = CO₂ × Δ¹⁴C with Δ¹⁴C a dimensionless fraction (per mil ÷ 1000),
the convention of Basu et al. (2016):

```
output/flux_c14.ocedis.<year>.nc          the reconstruction
output/flux_c14.ocedis_table1.<year>.nc   rescaled to the published annual total
    c14flux(time, lat, lon)   [umol m-2 s-1]
```

The second variant multiplies each year by one scalar so its annual global
isoflux matches Miller et al. (2025) Table 1 exactly. Amplitude only: the
pattern, the seasonality and the sign are the reconstruction's. Use it when a
downstream run should carry the published magnitude rather than this
reconstruction's independent one.

Land cells are written as `0.0` rather than NaN, because a NaN in an emission
field propagates into the tracer.

**A gridded atmospheric Δ¹⁴C field**, monthly, 1°×1°, 1990–2025, built from the
same station readers:

```
output/d14c_atm.<year>.nc       d14c, d14c_uncertainty (time, lat, lon)  [permil]
output/d14c_atm_timeseries.nc   the scalar series, for consumers that want one number
```

Per-station curve fits in time, then universal kriging with a sin(latitude)
drift in space, one month at a time. Cross-validation is the interesting part:
leave-one-station-out residual kriging improves RMSE by 0.1 %, so the network
constrains a linear interhemispheric gradient and essentially nothing else.
The field is close to zonally uniform by measurement rather than by assumption.

**A regional, hourly variant** at any resolution, for a bounded domain:

```bash
python scripts/make_regional_hourly.py --year 2024 \
    --resolution 0.1 --domain 33 73 -15 35 --tag eur010x010 \
    --atm-geometry krige --era5-dir data/processed/era5_hourly
```

The hourly structure comes from the wind alone. Because `k = γ⟨U²⟩(Sc/660)^-0.5`
is linear in ⟨U²⟩, driving it with hourly winds rather than a monthly mean
leaves the monthly mean of the hourly flux equal to the monthly reconstruction;
the script measures that identity rather than asserting it. Δoce, Δatm and pCO₂
are held constant within each month, which is their native resolution.

## Install

```bash
python3 -m pip install -e ".[dev]"
pytest tests/
```

Tests that need downloaded data skip automatically when it is absent, so this
works on a fresh clone. The suite exercises considerably more once the data
below is in place.

## Data you have to supply

None of the input data is redistributed here. Each source has its own licence,
and together they are several gigabytes.

| Under `data/raw/` | Source | Used for |
|---|---|---|
| `GLODAP/` | https://glodap.info | the 1995 surface Δ¹⁴C-DIC climatology |
| `Takahashi2009/` | NOAA OCADS | the monthly surface ocean pCO₂ climatology |
| `Delta_atm/` | NOAA GML, Scripps, ICOS, Heidelberg | the 22 atmospheric Δ¹⁴C station records |
| `Hua2021/` | published compilation | an alternative atmospheric Δ¹⁴C representation |
| `GLODAPv2.2020/` | https://glodap.info | repeat-section data, for the regression refit only |

Reanalysis winds are fetched and reduced on the fly:

```bash
python -m ocean14c.atmosphere_forcing              # global monthly <U^2>
python data/ERA5_hourly_regional.py --year 2024 \
    --domain 33 73 -15 35 --out-dir data/processed/era5_hourly
```

Both delete the raw download once reduced; u10 and v10 are only ever needed as
⟨U²⟩. Fetching needs a Copernicus Climate Data Store account and a `.cdsapirc`.

`docs/DATA.md` documents every format, including the parts that had to be
reverse-engineered.

## Running it

```bash
python3 scripts/validate_full_series.py            # the full series and its validation
python3 scripts/build_delta_atm_grid.py            # the gridded atmospheric field
python3 scripts/validate_delta_atm_grid.py         # its cross-validation and figures
```

`validate_full_series.py` writes the flux files by default, converts each year
back to PgC ‰ yr⁻¹ as a round-trip check, and aborts if the two disagree.

Three further scripts exist for questions that came up during development and
are worth keeping reproducible: `compare_delta_atm_variants.py` scores six ways
of building Δatm against each other, `compare_glodap_trends.py` tests the
rate-of-change regression against repeat-section observations, and
`refit_regression_glodap.py` refits it from a newer data version.

## Three choices the literature does not settle

These are config toggles in `config/default.yaml`, not implementation
shortcuts. Each is a genuine ambiguity, and `docs/SCIENCE.md` gives the full
reasoning.

**The tropical Δatm correction band.** A +10 ‰ adjustment appears in the
literature over two different latitude bands, 10°N–40°S and 20°N–20°S, for
what is nominally the same correction. Scored against the 22-station network by
leave-one-station-out, both lose to adding no tropical structure at all
(3.85 ‰ and 4.37 ‰ against 2.92 ‰). The band remains the default because every
shipped number was built with it, and the evidence against it comes from a
network that has only three sites inside it. The choice is worth up to 27 % of
the global isoflux.

**The regression coefficients.** The original set reproduces the published
tables exactly; a later data version is available as the documented sensitivity
check.

**The gas transfer calibration γ.** The leading coefficient of `k` is not
universal: it is fit to the global bomb-¹⁴C inventory for a specific wind
product, because it depends on which moment of the wind distribution is used.
The value appropriate to the reanalysis the reference used is not published.
This package defaults to γ = 0.31, which is at least structurally correct for a
true second moment ⟨U²⟩, and `ocean14c.validate.fit_gamma()` refits it against
the published one-way flux series.

## Documentation

| Doc | Covers |
|---|---|
| `docs/SCIENCE.md` | the method: every equation, constant and source, stage by stage |
| `docs/DATA.md` | where each dataset comes from, its format, and its traps |
| `docs/ARCHITECTURE.md` | module map, unit and dimension conventions, test layout |
| `docs/USAGE.md` | install, worked examples |
| `docs/DEVELOPMENT.md` | how to extend it, and what is left open |

## Layout

```
src/ocean14c/
  pipeline.py        reconstruct_year(), the whole chain
  regional.py        the bounded-domain, arbitrary-resolution, hourly variant
  glodap.py          the 1995 climatology and its region masks
  regression.py      the two-population rate-of-change regression
  surface_ocean.py   projecting Delta_oce forward from 1995
  atmosphere.py      Delta_atm, in six selectable geometries
  atmfield.py        the kriged station field
  atm_stations.py    readers for each station data format
  krige.py           universal kriging with a latitude drift
  curvefit.py        per-station curve fitting
  hua.py             the published-curve alternative
  gas_exchange.py    Schmidt number and the transfer velocity
  solubility.py      CO2 solubility in seawater
  foa.py             the one-way flux
  takahashi.py       the pCO2 climatology and its drift
  winds.py           wind forcing
  atmosphere_forcing.py   reanalysis download and reduction to <U^2>
  isoflux.py         combine and aggregate
  regions.py         the Southern Ocean isoline and the North Pacific split
  table1.py          the rescaled-to-published variant
  validate.py        comparison against the published tables
  output.py          NetCDF writers
  vendor/            a vendored curve-fitting implementation
scripts/             the runnable entry points above
config/default.yaml  every switchable choice
tests/               159 tests across 20 modules
```

## References

Miller, J. B., Lehman, S. J., & Lindsay, C. M. (2025). Numerical representation
of contemporary atmospheric Δ¹⁴CO₂: 1. Time-varying global fluxes and
atmospheric mass balance. *Global Biogeochemical Cycles*, 39, e2025GB008522.
https://doi.org/10.1029/2025GB008522

Key, R. M., et al. (2004). A global ocean carbon climatology: results from
GLODAP. *Global Biogeochemical Cycles*, 18, GB4031.
https://doi.org/10.1029/2004GB002247

Takahashi, T., et al. (2009). Climatological mean and decadal change in surface
ocean pCO₂, and net sea-air CO₂ flux over the global oceans. *Deep-Sea Research
II*, 56, 554-577. https://doi.org/10.1016/j.dsr2.2008.12.009

Wanninkhof, R. (1992). Relationship between wind speed and gas exchange over
the ocean. *Journal of Geophysical Research*, 97, 7373–7382.
https://doi.org/10.1029/92JC00188

Wanninkhof, R. (2014). Relationship between wind speed and gas exchange over
the ocean revisited. *Limnology and Oceanography: Methods*, 12, 351–362.
https://doi.org/10.4319/lom.2014.12.351

Sweeney, C., et al. (2007). Constraining global air-sea gas exchange for CO₂
with recent bomb ¹⁴C measurements. *Global Biogeochemical Cycles*, 21, GB2015.
https://doi.org/10.1029/2006GB002784

Basu, S., Miller, J. B., & Lehman, S. (2016). Separation of biospheric and
fossil fuel fluxes of CO₂ by atmospheric inversion of CO₂ and ¹⁴CO₂
measurements. *Atmospheric Chemistry and Physics*, 16, 5665–5683.
https://doi.org/10.5194/acp-16-5665-2016

Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of
the Royal Meteorological Society*, 146, 1999–2049.
https://doi.org/10.1002/qj.3803

Stuiver, M., & Polach, H. A. (1977). Discussion: Reporting of ¹⁴C data.
*Radiocarbon*, 19(3), 355–363.

## Citation

If you use this code, please also cite the paper whose method it reconstructs,
Miller et al. (2025) above.

## Author

Carlos Gómez-Ortiz, Department of Earth and Environmental Sciences, Lund
University. carlos.gomez@mgeo.lu.se

## License

Not yet chosen. The input data under `data/raw/` is third-party, is not
redistributed here, and carries its own licence per source; see `docs/DATA.md`.
