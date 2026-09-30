"""End-to-end reconstruction for one year: composes every pipeline stage
(docs/SCIENCE.md) into annual global / Southern Ocean isoflux numbers,
directly comparable to Table 1 / Table S2 (ocean14c.validate) and the
paper's own gridded product (Ocediseq.nc at the repo root).

This is the orchestration piece flagged as missing in docs/ARCHITECTURE.md
and docs/DEVELOPMENT.md -- every module it calls was already independently
tested; what's new here is wiring them together and getting the monthly
vs. annual aggregation convention right (see the module docstring of
ocean14c.isoflux for the straight-sum vs. flux-weighted distinction).
"""

from pathlib import Path

import numpy as np
import xarray as xr

from ocean14c.atm_stations import load_c14_station, load_drp_curl_event
from ocean14c import hua
from ocean14c.atmosphere import (
    DEFAULT_CURVE_FIT,
    NORTHERN_HEMISPHERE_WINDOWS,
    SOUTHERN_HEMISPHERE_WINDOWS,
    area_weighted_mean,
    delta_atm_field,
    delta_atm_zonal,
    extrapolation_gap,
    fit_year_edge_values,
    relevel,
    select_station,
)
from ocean14c.foa import compute_foa, foa_flux_to_pgc_per_yr
from ocean14c.gas_exchange import DEFAULT_GAMMA
from ocean14c.glodap import surface_delta_1995
from ocean14c.isoflux import compute_isoflux, flux_weighted_mean, spatial_sum, to_umol_flux
from ocean14c.regions import southern_ocean_mask
from ocean14c.surface_ocean import project_delta_oce
from ocean14c.takahashi import load_pco2_climatology, load_sst_climatology, project_pco2_ocean, regrid_nearest
from ocean14c.winds import load_u2

EARTH_RADIUS_M = 6_371_000.0
MONTHS = np.arange(1, 13)


def cell_area_m2(lat):
    """Approximate area [m^2] of a 1x1 degree grid cell at `lat`."""
    dlat = np.deg2rad(1.0)
    dlon = np.deg2rad(1.0)
    return (EARTH_RADIUS_M**2) * dlon * dlat * np.cos(np.deg2rad(lat))


def _load_atm_station_series(delta_atm_dir):
    d = Path(delta_atm_dir)
    nh = {
        "NWR": load_c14_station(d / "noaa_l2_2025_1_nwr_3m_spot_min_clean.c14"),
        "JFJ": load_c14_station(d / "uheicrl_l2_2025_1_jfj_5m_int_14day_clean.c14"),
    }
    sh = {
        "SPO": load_c14_station(d / "ucsd_l2_2025_1_spo_2m_spot_1min_clean.c14"),
        "PSA": load_c14_station(d / "ucsd_l2_2025_1_psa_10m_spot_min_clean.c14"),
        "SMO": load_c14_station(d / "ucsd_l2_2025_1_smo_2m_spot_1min_clean.c14"),
        "DRP": load_drp_curl_event(d / "co2c14_drp_shipboard-flask_376_curl_event.txt"),
    }
    return nh, sh


def _monthly_hemispheric_series(station_series, windows, year,
                                curve_fit=DEFAULT_CURVE_FIT):
    """Evaluate one hemisphere's spliced curve fit at all 12 month midpoints
    of `year`, using whichever station's window covers that year.

    Returns (values, station_name, extrapolation_gap_yr). The gap is 0.0 when
    the year is fully covered by observations; see atmosphere.extrapolation_gap.
    """
    station = select_station(windows, year)
    if station is None:
        raise ValueError(
            f"no station window covers year {year} -- see "
            f"ocean14c.atmosphere's window tables for the supported range"
        )
    series = station_series[station]
    decimal_months = year + (MONTHS - 0.5) / 12
    edges = fit_year_edge_values(series, decimal_months.tolist(), method=curve_fit)
    return (
        np.array([edges[t] for t in decimal_months]),
        station,
        extrapolation_gap(series, year),
    )


def _monthly_delta_oce(delta_1995, year, regression_version="original"):
    decimal_months = year + (MONTHS - 0.5) / 12
    slices = [
        project_delta_oce(delta_1995, t, version=regression_version)
        for t in decimal_months
    ]
    return xr.concat(slices, dim=xr.DataArray(MONTHS, dims="month", name="month"))


def _monthly_delta_atm(lat, nh_monthly, sh_monthly):
    slices = [
        delta_atm_field(lat, delta_n=nh_monthly[i], delta_s=sh_monthly[i]) for i in range(12)
    ]
    return xr.concat(slices, dim=xr.DataArray(MONTHS, dims="month", name="month"))


# ---------------------------------------------------------------------------
# Alternative atmospheric forcings, for scripts/compare_delta_atm_variants.py.
#
# atm_variant=None keeps _monthly_delta_atm above, byte for byte, so the pinned
# Table 1 / Table S2 / Ocediseq reproduction cannot move. Every alternative is
# opt-in.
#
# A variant is {"geometry": ..., "level": ..., "blend_deg": ...}:
#
#   geometry "miller_band"  NH/SH + 10 permil over 10N..40S -- identical to the
#                           default path; the regression guard.
#            "basu_band"    NH/SH + 10 permil over 20N..20S
#            "hemispheric"  plain NH/SH step at the Equator (what Hua et al.
#                           2021 reduces to for 1973 onward)
#            "hua"          Hua's own published curves, not our stations
#            "krige"        the shipped gridded field, output/d14c_atm.<year>.nc
#   level    "native" (default) or "stations" -- the latter re-levels a profile
#            onto the global mean of our own hemispheric series, which is only
#            meaningful for geometry="hua"
#   blend_deg  half-width of a linear blend across internal band boundaries
# ---------------------------------------------------------------------------

DEFAULT_ATM_VARIANT = {"geometry": "miller_band", "level": "native", "blend_deg": 0.0}


def _kriged_delta_atm(lat, lon, year, krige_dir="output"):
    """Monthly (month, lat, lon) Delta14C from the shipped gridded product.

    Unlike every other variant this is not zonally uniform, and it is the only
    one carrying its own per-cell uncertainty. It is also the only one already
    fitted to the full 22-station network rather than the six sites the
    hemispheric splice uses.
    """
    import netCDF4 as nc

    path = Path(krige_dir) / f"d14c_atm.{year}.nc"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- build it with "
            f"scripts/build_delta_atm_grid.py --start {year} --end {year}"
        )
    with nc.Dataset(path) as ds:
        field = np.asarray(ds.variables["d14c"][:], dtype=float)   # (12, nlat, nlon)
        flat = np.asarray(ds.variables["lat"][:], dtype=float)
        flon = np.asarray(ds.variables["lon"][:], dtype=float)

    da = xr.DataArray(
        field,
        dims=("month", "lat", "lon"),
        coords={"month": MONTHS, "lat": flat, "lon": flon},
    )
    if not (np.array_equal(flat, np.asarray(lat)) and np.array_equal(flon, np.asarray(lon))):
        da = da.interp(lat=lat, lon=lon)
    return da


def build_delta_atm(lat, lon, year, nh_monthly, sh_monthly, variant=None,
                    krige_dir="output", hua_series=None):
    """Monthly Delta_atm for one year under `variant`. See the note above."""
    if variant is None:
        return _monthly_delta_atm(lat, nh_monthly, sh_monthly)

    spec = dict(DEFAULT_ATM_VARIANT)
    spec.update(variant)
    geometry, level, blend = spec["geometry"], spec["level"], spec["blend_deg"]

    if geometry == "krige":
        if level != "native":
            raise ValueError(
                'geometry="krige" has no level to choose -- it IS our station '
                "network at our own level"
            )
        return _kriged_delta_atm(lat, lon, year, krige_dir)

    lat_arr = np.asarray(lat)
    decimal_months = year + (MONTHS - 0.5) / 12

    slices = []
    for i, t in enumerate(decimal_months):
        if geometry == "hua":
            profile = hua.field_at(lat_arr, float(t), series=hua_series,
                                   blend_deg=blend)
            if level == "stations":
                # Hua's shape, our level. The two differ by ~6-8 permil in
                # absolute terms while agreeing on the interhemispheric
                # gradient, and Delta_atm enters as Delta_source - Delta_atm,
                # so the choice is not cosmetic.
                ours = delta_atm_zonal(
                    lat_arr, {"NH": nh_monthly[i], "SH": sh_monthly[i]}, "hemispheric"
                )
                profile = relevel(lat_arr, profile, area_weighted_mean(lat_arr, ours))
            elif level != "native":
                raise ValueError(f"unknown level: {level!r}")
        else:
            if level != "native":
                raise ValueError(
                    f'level={level!r} applies only to geometry="hua"; '
                    f"{geometry!r} is already built from our own stations"
                )
            profile = delta_atm_zonal(
                lat_arr, {"NH": nh_monthly[i], "SH": sh_monthly[i]},
                geometry, blend_deg=blend,
            )
        slices.append(xr.DataArray(profile, coords=lat.coords, dims=lat.dims)
                      if isinstance(lat, xr.DataArray)
                      else xr.DataArray(profile, dims="lat"))

    return xr.concat(slices, dim=xr.DataArray(MONTHS, dims="month", name="month"))


def reconstruct_year(
    year: int,
    data_dir="data",
    gamma: float = DEFAULT_GAMMA,
    sc_poly: str = "1992",
    regression_version: str = "original",
    curve_fit: str = DEFAULT_CURVE_FIT,
    atm_variant: dict = None,
    krige_dir: str = "output",
    hua_series=None,
) -> dict:
    """Reconstruct annual global / Southern Ocean isoflux for one year.

    Returns a dict with the monthly per-cell isoflux DataArray plus scalar
    summaries: global_isoflux (PgC permil/yr, straight sum -- Table 1's
    aggregator), global_foa / global_delta_odis / socean_foa /
    socean_delta_odis (Table S2's aggregators: Foa is a straight spatial
    sum of the annual-mean field, Delta_odis is flux-weighted).

    Also returns c14flux_monthly: the same isoflux as a per-area C_D14C flux
    [umol m^-2 s^-1], dims (month, lat, lon) -- the TM5-facing product, written
    to disk by ocean14c.output.write_year(). The PgC permil/yr diagnostics above
    are retained unchanged so the Table 1 / Table S2 / Ocediseq validation keeps
    comparing against the quantity those tables actually report.
    """
    data_dir = Path(data_dir)
    glodap_dir = data_dir / "raw" / "GLODAP"
    taka_path = data_dir / "raw" / "Takahashi2009" / "month_flux_2006c.txt"
    delta_atm_dir = data_dir / "raw" / "Delta_atm"
    era5_path = data_dir / "processed" / "era5_u2" / f"u2_{year}.nc"

    delta_1995 = surface_delta_1995(glodap_dir)
    lat, lon = delta_1995["lat"], delta_1995["lon"]

    delta_oce_monthly = _monthly_delta_oce(delta_1995, year, regression_version)

    nh_series, sh_series = _load_atm_station_series(delta_atm_dir)
    nh_monthly, nh_station, nh_gap = _monthly_hemispheric_series(
        nh_series, NORTHERN_HEMISPHERE_WINDOWS, year, curve_fit=curve_fit
    )
    sh_monthly, sh_station, sh_gap = _monthly_hemispheric_series(
        sh_series, SOUTHERN_HEMISPHERE_WINDOWS, year, curve_fit=curve_fit
    )
    delta_atm_monthly = build_delta_atm(
        lat, lon, year, nh_monthly, sh_monthly,
        variant=atm_variant, krige_dir=krige_dir, hua_series=hua_series,
    )

    delta_odis_monthly = delta_oce_monthly - delta_atm_monthly

    clim = load_pco2_climatology(taka_path)
    sst_clim = load_sst_climatology(taka_path)
    pco2_1x1 = regrid_nearest(clim, lat, lon)
    sst_1x1 = regrid_nearest(sst_clim, lat, lon)
    pco2_ocean_monthly = project_pco2_ocean(pco2_1x1, year=year)

    u2_monthly = load_u2(era5_path)
    u2_1x1 = u2_monthly.interp(lat=lat, lon=lon, method="nearest")

    foa_flux_monthly = compute_foa(u2_1x1, sst_1x1, pco2_ocean_monthly, gamma=gamma, sc_poly=sc_poly)
    area = cell_area_m2(lat)
    foa_pgc_per_yr_monthly = foa_flux_to_pgc_per_yr(foa_flux_monthly, area)

    isoflux_monthly = compute_isoflux(foa_pgc_per_yr_monthly, delta_odis_monthly)
    annual_isoflux = isoflux_monthly.mean(dim="month")  # PgC permil/yr, annual-mean rate
    annual_foa = foa_pgc_per_yr_monthly.mean(dim="month")

    # TM5-facing product: the same isoflux expressed per unit area, per second,
    # with Delta as a dimensionless fraction (see ocean14c.isoflux.to_umol_flux).
    # Built from the per-area foa_flux_monthly rather than the per-cell PgC/yr
    # field, so it carries no cell-area factor.
    c14flux_monthly = to_umol_flux(foa_flux_monthly, delta_odis_monthly)

    so_mask = southern_ocean_mask(delta_1995)

    return {
        "isoflux_monthly": isoflux_monthly,
        "annual_isoflux": annual_isoflux,
        "c14flux_monthly": c14flux_monthly,
        "global_isoflux": spatial_sum(annual_isoflux),
        "global_foa": spatial_sum(annual_foa),
        "global_delta_odis": flux_weighted_mean(delta_odis_monthly, foa_pgc_per_yr_monthly),
        "socean_foa": spatial_sum(annual_foa, mask=so_mask),
        "socean_delta_odis": flux_weighted_mean(delta_odis_monthly, foa_pgc_per_yr_monthly, mask=so_mask),
        # Provenance for the atmospheric forcing: which station fed each
        # hemisphere, and how far past its last observation this year sits
        # (0.0 = fully observed). See docs/SCIENCE.md Sec 9.
        "atm_station_nh": nh_station,
        "atm_station_sh": sh_station,
        "atm_extrapolation_yr": max(nh_gap, sh_gap),
        "curve_fit": curve_fit,
        "atm_variant": dict(DEFAULT_ATM_VARIANT) if atm_variant is None
                       else {**DEFAULT_ATM_VARIANT, **atm_variant},
        "delta_atm_monthly": delta_atm_monthly,
    }
