"""Build the gridded atmospheric Delta14C field from the station network.

Pipeline:
    stations.load_stations   -> registry with coordinates from file headers
    background.filter_all    -> ccgcrv fit + one-sided outlier rejection
    krige.krige_month        -> one month at a time onto a lat/lon grid
    write_year / write_timeseries

Output is a monthly (time, lat, lon) field in permil with a companion
uncertainty field, plus the scalar Delta14C(year, month) series in the format
TM5's `atmospheric.D14C.timeseries.output` rc-key expects today (see
tm5.radiocarbon_emissions.prepare_atmos_del_timeseries), so the existing
scalar workflow keeps working unchanged.
"""

import datetime
import warnings
import os

import netCDF4 as nc
import numpy as np

from ocean14c.background import filter_all
from ocean14c.curvefit import month_midpoints
from ocean14c.krige import (
    DEFAULT_DRIFT_DEGREE,
    DEFAULT_L_LAT_KM,
    DEFAULT_L_LON_KM,
    DEFAULT_OBS_ERR,
    DEFAULT_SILL,
    REGIONAL_OBS_ERR,
    krige_month,
)
from ocean14c.output import month_start_days
from ocean14c.stations import load_stations

# Monthly fields are stamped at the START of the month -- the archive-wide
# convention. Imported from the flux writer so there is one definition.

# Below this many contributing stations a month is not written: the drift is a
# 2-parameter fit, so 4 is already thin and anything less is meaningless.
MIN_STATIONS = 4


def default_grid(res_deg=1.0):
    """Cell-centre lat/lon arrays, matching the convention used elsewhere in
    this package."""
    nlat = int(round(180.0 / res_deg))
    nlon = int(round(360.0 / res_deg))
    lat = np.linspace(-90 + res_deg / 2, 90 - res_deg / 2, nlat)
    lon = np.linspace(-180 + res_deg / 2, 180 - res_deg / 2, nlon)
    return lat, lon


# Extra 1-sigma error per year of extrapolation beyond a station's record.
# From the hold-out test in curvefit.StationFit.extrapolate: a 2-year
# extrapolation has total RMSE 2.78 permil against a within-record station
# scatter of ~1.5, so the extrapolation-specific part is
# sqrt(2.78^2 - 1.5^2) / 2 ~ 1.2 permil per year.
EXTRAP_ERR_PER_YEAR = 1.2


def station_values_at(stations, fits, t, obs_err=DEFAULT_OBS_ERR,
                      regional_obs_err=REGIONAL_OBS_ERR,
                      max_extrapolation_yr=0.0):
    """Station lat/lon/value/error at decimal date `t`.

    By default only stations whose observed record covers `t` contribute --
    ccgcrv is a smoother, and `getSmoothValue` returns NaN outside the record.

    max_extrapolation_yr > 0 additionally admits stations whose record ends up
    to that many years before `t`, using
    curvefit.StationFit.evaluate (trend continued at its end-of-record growth
    rate, plus the seasonal cycle). Their observation error is inflated by
    EXTRAP_ERR_PER_YEAR per year of gap, so the kriging downweights them and
    the uncertainty field reflects the extra risk.
    """
    lat, lon, val, err = [], [], [], []
    for code, fit in fits.items():
        gap = float(fit.gap(t)[0])
        if gap > max_extrapolation_yr:
            continue
        st = stations[code]
        value = float(fit.evaluate(t)[0]) if gap > 0 else float(fit.smooth(t)[0])
        if not np.isfinite(value):
            continue
        base = obs_err if st.is_background else regional_obs_err
        lat.append(st.lat)
        lon.append(st.lon)
        val.append(value)
        err.append(np.hypot(base, gap * EXTRAP_ERR_PER_YEAR))
    return (np.array(lat), np.array(lon), np.array(val), np.array(err))


def build_field(
    stations, fits, year_start, year_end,
    res_deg=1.0,
    l_lat_km=DEFAULT_L_LAT_KM,
    l_lon_km=DEFAULT_L_LON_KM,
    sill=DEFAULT_SILL,
    drift_degree=DEFAULT_DRIFT_DEGREE,
    obs_err=DEFAULT_OBS_ERR,
    min_stations=MIN_STATIONS,
    max_extrapolation_yr=0.0,
):
    """Grid every month in [year_start, year_end].

    Returns dict with: lat, lon, times (decimal), years, months,
    field (nt, nlat, nlon), sd (nt, nlat, nlon), n_stations (nt,).
    Months with fewer than `min_stations` contributors are NaN.
    """
    grid_lat, grid_lon = default_grid(res_deg)
    times = month_midpoints(year_start, year_end)
    years = np.repeat(np.arange(year_start, year_end + 1), 12)
    months = np.tile(np.arange(1, 13), year_end - year_start + 1)

    nt = len(times)
    field = np.full((nt, len(grid_lat), len(grid_lon)), np.nan)
    sd = np.full_like(field, np.nan)
    n_st = np.zeros(nt, dtype=int)

    for i, t in enumerate(times):
        la, lo, va, er = station_values_at(
            stations, fits, t, obs_err=obs_err,
            max_extrapolation_yr=max_extrapolation_yr,
        )
        n_st[i] = len(va)
        if len(va) < min_stations:
            continue
        f, s = krige_month(
            la, lo, va, er, grid_lat, grid_lon,
            l_lat_km=l_lat_km, l_lon_km=l_lon_km,
            sill=sill, drift_degree=drift_degree,
        )
        field[i] = f
        sd[i] = s

    return {
        "lat": grid_lat, "lon": grid_lon, "times": times,
        "years": years, "months": months,
        "field": field, "sd": sd, "n_stations": n_st,
        "max_extrapolation_yr": max_extrapolation_yr,
    }


def zonal_band_mean(result, lat_south, lat_north):
    """Area-weighted mean of the field over a latitude band, per month.

    Returns NaN for months with no valid data anywhere in the band, and
    renormalises the cos(lat) weights over the valid cells only.

    The NaN handling is load-bearing, not defensive: `np.nansum` of an
    all-NaN slice returns 0.0, not NaN. An earlier version used it directly
    and wrote Delta14C = 0.000 permil into the TM5 timeseries for 2024-2025,
    where too few stations remain to grid a field at all -- a silently wrong
    forcing rather than an obviously missing one.
    """
    lat = result["lat"]
    m = (lat >= lat_south) & (lat <= lat_north)
    w = np.cos(np.radians(lat[m]))
    band = result["field"][:, m, :]

    with warnings.catch_warnings():
        # An all-NaN row is expected for months with too few stations; the
        # NaN it produces is exactly what we want to propagate.
        warnings.simplefilter("ignore", RuntimeWarning)
        zonal = np.nanmean(band, axis=2)            # (nt, nband), NaN where empty
    valid = np.isfinite(zonal)
    wsum = (valid * w[None, :]).sum(axis=1)
    num = np.nansum(np.where(valid, zonal * w[None, :], 0.0), axis=1)

    out = np.full(len(num), np.nan)
    np.divide(num, wsum, out=out, where=wsum > 0)
    return out


def global_mean_series(result):
    """Area-weighted global mean Delta14C per month [permil]."""
    return zonal_band_mean(result, -90.0, 90.0)


def write_year(result, year, out_dir, compress=True):
    """Write one year of the gridded field to d14c_atm.<year>.nc."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"d14c_atm.{year}.nc")

    sel = result["years"] == year
    if sel.sum() != 12:
        raise ValueError(f"expected 12 months for {year}, got {int(sel.sum())}")

    lat, lon = result["lat"], result["lon"]
    zlib = compress
    clevel = 4 if compress else 0

    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", 12)
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))

        t = ds.createVariable("time", "f8", ("time",))
        t.units = f"days since {year}-01-01 00:00:00"
        t.calendar = "standard"
        t.long_name = "time"
        t.standard_name = "time"
        t.axis = "T"
        t[:] = np.array(month_start_days(year), dtype="f8")

        v = ds.createVariable("lat", "f4", ("lat",))
        v.units, v.long_name, v.standard_name = "degrees_north", "latitude", "latitude"
        v[:] = lat.astype("f4")

        v = ds.createVariable("lon", "f4", ("lon",))
        v.units, v.long_name, v.standard_name = "degrees_east", "longitude", "longitude"
        v[:] = lon.astype("f4")

        d = ds.createVariable("d14c", "f4", ("time", "lat", "lon"),
                              zlib=zlib, complevel=clevel)
        d.units = "permil"
        d.long_name = "Atmospheric Delta14C in CO2"
        d.comment = (
            "Background field: fossil-affected sites are downweighted and "
            "episodic pollution rejected, because TM5 simulates fossil "
            "dilution separately via CO2_ff x fossil_del = -1.0. Delta14C is "
            "in permil here; TM5's C_D14C tracer uses the dimensionless "
            "fraction (permil/1000)."
        )
        d[:] = result["field"][sel].astype("f4")

        u = ds.createVariable("d14c_uncertainty", "f4", ("time", "lat", "lon"),
                              zlib=zlib, complevel=clevel)
        u.units = "permil"
        u.long_name = "Kriging standard deviation"
        u.comment = (
            "1-sigma prediction uncertainty from the universal-kriging system. "
            "Large where no station constrains the field."
        )
        u[:] = result["sd"][sel].astype("f4")

        n = ds.createVariable("n_stations", "i4", ("time",))
        n.long_name = "Number of stations contributing to this month"
        n[:] = result["n_stations"][sel].astype("i4")

        n_valid = int(sum(np.isfinite(result["field"][sel][i]).any()
                          for i in range(12)))
        ds.title = "Gridded atmospheric Delta14C from the surface station network"
        ds.year = str(year)
        ds.n_valid_months = n_valid
        ds.max_extrapolation_yr = float(result.get("max_extrapolation_yr", 0.0))
        if n_valid < 12:
            ds.incomplete_year = (
                f"only {n_valid} of 12 months could be gridded; the rest have "
                f"fewer than {MIN_STATIONS} stations and are NaN"
            )
        ds.method = (
            "Per-station ccgcrv curve fit (Thoning et al. 1989) in time; "
            "universal kriging with a sin(latitude) drift in space, one month "
            "at a time."
        )
        ds.created = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return path


def write_timeseries(result, out_path):
    """Write the scalar Delta14C(year, month) series TM5 reads today.

    Format matches tm5.radiocarbon_emissions.prepare_atmos_del_timeseries:
    a (year, month) tuple index plus the value as a DIMENSIONLESS fraction
    (permil / 1000), which is the convention the Fortran expects.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    g = global_mean_series(result)
    ok = np.isfinite(g)

    years = result["years"][ok].astype("i2")
    months = result["months"][ok].astype("i2")
    values = (g[ok] / 1000.0).astype("f8")

    with nc.Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("n", len(values))
        ds.createDimension("two", 2)

        ym = ds.createVariable("year_month", "i2", ("n", "two"))
        ym.long_name = "year and month index"
        ym[:, 0] = years
        ym[:, 1] = months

        v = ds.createVariable("D14C", "f8", ("n",))
        v.units = "1"
        v.long_name = "Global mean atmospheric Delta14C, dimensionless fraction"
        v.comment = "permil / 1000, matching TM5's C_D14C convention"
        v[:] = values

        ds.title = "Global mean atmospheric Delta14C timeseries for TM5"
        ds.source = "Area-weighted global mean of the gridded d14c_atm field"
        ds.created = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return out_path


def build(delta_atm_dir, year_start, year_end, include_regional=False, **kwargs):
    """Convenience: registry -> background fits -> gridded field."""
    stations = load_stations(delta_atm_dir, include_regional=include_regional)
    fits, reports = filter_all(stations, include_regional=include_regional)
    result = build_field(stations, fits, year_start, year_end, **kwargs)
    result["reports"] = reports
    result["stations"] = stations
    result["fits"] = fits
    return result
