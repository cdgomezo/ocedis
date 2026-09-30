"""Atmospheric Delta14C field, Delta_atm(lat, t) (CLAUDE.md Sec 3.4).

Each station's raw record is smoothed with a simplified Thoning et al.
(1989)-style curve (polynomial trend + annual/semiannual harmonics fit by
least squares) to extract year-edge (Jan 1) values; per-station series are
spliced into a single hemispheric series over the analysis window, then a
zonally uniform tropical correction band is applied to get Delta_atm(lat,t).
"""

import numpy as np
import xarray as xr

# Station windows, in precedence order: the FIRST station whose window covers a
# year is the one used for it. A window is either a single (start, end) range or
# a tuple of ranges, so one station can serve two separate periods.
#
# 2000-2012 -- the Miller et al. (2025) validation window -- resolves exactly as
# before: NH JFJ 2000-2003 then NWR 2004-2012; SH SPO 1999-2007 then DRP
# 2008-2012 (SPO wins 1999-2007 on precedence). Do not reorder these dicts.
#
# 2013 onward extends the same splicing rule to the full length of each record
# (docs/SCIENCE.md Sec 9). The governing constraint is that a station is never
# asked for months beyond its last observation, since the curve fit is a global
# quadratic and extrapolates poorly:
#
#   station  record ends   last fully-covered year (needs year + 11.5/12)
#   NWR      2021.40       2020
#   JFJ      2024.33       2023
#   SPO      2024.00       2023
#   SMO      2023.95       2022
#   DRP      2016.96       2016
#
# NWR -> JFJ is a near-seamless splice: the two agree to -0.08 permil (sd 1.23)
# over 19 overlapping years. DRP -> SPO carries a ~1.3 permil step (DRP sits
# above SPO), which is within the ~2 permil spread among the SH stations.
#
# Years 2024-2025 necessarily extrapolate past every record; they are produced
# but flagged. See atmosphere.extrapolation_gap() and reconstruct_year()'s
# "atm_extrapolation_yr" output.
NORTHERN_HEMISPHERE_WINDOWS = {
    "NWR": (2004, 2020),
    "JFJ": ((2000, 2003), (2021, 2030)),
}
SOUTHERN_HEMISPHERE_WINDOWS = {
    "SPO": ((1999, 2007), (2017, 2030)),
    "PSA": (2005, 2007),
    "SMO": (2001, 2007),
    "DRP": (2006, 2016),
}

# Last year whose 12 month-midpoints are fully covered by observations, given
# the record ends above. Beyond this the atmospheric curve is extrapolated.
LAST_OBSERVED_YEAR = 2023


def window_covers(window, year) -> bool:
    """True if `year` falls in `window`, which is either a single (start, end)
    range or a tuple of them."""
    ranges = window if isinstance(window[0], (tuple, list)) else (window,)
    return any(start <= year <= end for start, end in ranges)


def select_station(windows: dict, year):
    """First station in `windows` whose window covers `year`, else None."""
    return next((s for s, w in windows.items() if window_covers(w, year)), None)


def extrapolation_gap(series, year) -> float:
    """Years by which `year`'s December midpoint falls past the end of `series`.

    0.0 when the year is fully covered by observations. The curve fit is a
    global quadratic, so a non-zero gap means the value is extrapolated and
    the decline rate is not observationally constrained there.
    """
    last_obs = max(t for t, _ in series)
    return max(0.0, (year + 11.5 / 12.0) - last_obs)

# Miller et al. (2025) main text default: +10 permil between 10N and 40S.
# Basu et al. (2016) alternate: +10 permil between 20N and 20S.
TROPICAL_BAND_MILLER = {"band_north": 10.0, "band_south": -40.0, "band_offset": 10.0}
TROPICAL_BAND_BASU = {"band_north": 20.0, "band_south": -20.0, "band_offset": 10.0}


def _design_matrix(t, poly_order=2, n_harmonics=2):
    columns = [np.ones_like(t)]
    for p in range(1, poly_order + 1):
        columns.append(t**p)
    for k in range(1, n_harmonics + 1):
        columns.append(np.sin(2 * np.pi * k * t))
        columns.append(np.cos(2 * np.pi * k * t))
    return np.column_stack(columns)


CURVE_FIT_SIMPLIFIED = "simplified"
CURVE_FIT_CCGCRV = "ccgcrv"
DEFAULT_CURVE_FIT = CURVE_FIT_SIMPLIFIED

# Cache fitted ccgcrv curves: reconstruct_year() re-fits the same station for
# every year of a long run, and the FFT filter is not cheap.
_CCGCRV_CACHE = {}


def _fit_year_edge_values_simplified(series, years, poly_order=2, n_harmonics=2) -> dict:
    """Bare least-squares polynomial + harmonics, no filtered residual.

    This is the original implementation and remains the DEFAULT, because the
    validated 2000-2012 reproduction (Table 1 / Table S2 / Ocediseq) was
    established with it. See fit_year_edge_values for the alternative.
    """
    t = np.array([d for d, _ in series], dtype=float)
    v = np.array([x for _, x in series], dtype=float)

    t0 = t.mean()
    design = _design_matrix(t - t0, poly_order=poly_order, n_harmonics=n_harmonics)
    coeffs, *_ = np.linalg.lstsq(design, v, rcond=None)

    years_arr = np.array(years, dtype=float)
    design_years = _design_matrix(years_arr - t0, poly_order=poly_order, n_harmonics=n_harmonics)
    fitted = design_years @ coeffs
    return dict(zip(years, fitted.tolist()))


def _fit_year_edge_values_ccgcrv(series, years) -> dict:
    """NOAA ccgcrv (Thoning et al. 1989), the method the papers actually cite.

    Adds the low-pass-filtered residual the simplified fit omits, and
    extrapolates beyond the record by continuing the trend at its end-of-record
    growth rate plus the seasonal cycle (ocean14c.curvefit.StationFit.evaluate).
    """
    from ocean14c.curvefit import fit_station

    key = (len(series), series[0][0], series[-1][0], series[-1][1])
    fit = _CCGCRV_CACHE.get(key)
    if fit is None:
        fit = fit_station("series", series)
        if fit is None:
            # Too short/sparse for ccgcrv; fall back rather than fail.
            return _fit_year_edge_values_simplified(series, years)
        _CCGCRV_CACHE[key] = fit

    values = fit.evaluate(np.array(list(years), dtype=float))
    return dict(zip(years, [float(v) for v in values]))


def fit_year_edge_values(series, years, poly_order=2, n_harmonics=2,
                         method=DEFAULT_CURVE_FIT) -> dict:
    """Fit a smooth curve to `series` ((decimal_date, value) pairs) and
    evaluate it at each decimal date in `years`.

    method:
      "simplified" (default) -- polynomial + harmonics by least squares.
      "ccgcrv"               -- NOAA ccgcrv, which additionally low-pass
                                filters the residual and can extrapolate
                                past the end of a record.

    The default is deliberately the simplified fit: it is what the validated
    2000-2012 reproduction was built on, so switching the default would move
    numbers that are pinned against Table 1 / Table S2 / Ocediseq.nc. See
    docs/SCIENCE.md Sec 11 for the measured effect of switching.
    """
    if method == CURVE_FIT_CCGCRV:
        return _fit_year_edge_values_ccgcrv(series, years)
    if method == CURVE_FIT_SIMPLIFIED:
        return _fit_year_edge_values_simplified(
            series, years, poly_order=poly_order, n_harmonics=n_harmonics
        )
    raise ValueError(f"unknown curve-fit method: {method!r}")


def splice_station_series(station_series: dict, windows: dict, years) -> dict:
    """Year-edge hemispheric series, picking whichever station's window
    (inclusive, [start, end]) covers each year."""
    result = {}
    for year in years:
        station = select_station(windows, year)
        if station is None:
            raise ValueError(f"no station window covers year {year}")
        edge = fit_year_edge_values(station_series[station], [year])
        result[year] = edge[year]
    return result


def delta_atm_field(
    lat: xr.DataArray,
    delta_n: float,
    delta_s: float,
    band_north: float = TROPICAL_BAND_MILLER["band_north"],
    band_south: float = TROPICAL_BAND_MILLER["band_south"],
    band_offset: float = TROPICAL_BAND_MILLER["band_offset"],
) -> xr.DataArray:
    """Zonally uniform Delta_atm(lat) for one year: hemispheric base value
    (Delta14CN for lat>=0, Delta14CS for lat<0) plus the tropical correction
    band."""
    base = xr.where(lat >= 0, delta_n, delta_s)
    in_band = (lat <= band_north) & (lat >= band_south)
    return base + xr.where(in_band, band_offset, 0.0)


# ---------------------------------------------------------------------------
# Zonal geometries, for scoring one representation of Delta_atm(lat) against
# another (scripts/validate_delta_atm_grid.py, scripts/compare_delta_atm_variants.py).
#
# delta_atm_field above is untouched and remains what the ocedis pipeline
# calls, because the pinned Table 1 / Table S2 / Ocediseq reproduction was
# established with it. Everything below is additive.
#
# A geometry is a partition of latitude into bands, plus an optional additive
# offset over a further band. The candidates:
#
#   hemispheric  NH/SH split at the Equator. What Hua et al. (2021) reduces to
#                for 1973 onward, when they pool records by hemisphere.
#   miller_band  hemispheric + 10 permil over 10N..40S (Miller et al. 2025).
#                Identical to delta_atm_field; this is what ocedis uses today.
#   basu_band    hemispheric + 10 permil over 20N..20S (Basu et al. 2016).
#   graven       three bands at 30N / 30S (Graven et al. 2017 CMIP6 sectors).
#
# Two more candidates are NOT band geometries and live elsewhere: the linear
# sin(lat) drift (krige.drift_basis) and the shipped kriged field
# (atmfield.build_field). Both are scored in the same ladder.
# ---------------------------------------------------------------------------

BAND_EDGES = {
    "hemispheric": (("SH", -90.0, 0.0), ("NH", 0.0, 90.0)),
    "miller_band": (("SH", -90.0, 0.0), ("NH", 0.0, 90.0)),
    "basu_band": (("SH", -90.0, 0.0), ("NH", 0.0, 90.0)),
    "graven": (("SH", -90.0, -30.0), ("TR", -30.0, 30.0), ("NH", 30.0, 90.0)),
}

BAND_OFFSETS = {
    "miller_band": TROPICAL_BAND_MILLER,
    "basu_band": TROPICAL_BAND_BASU,
}


def band_of(lat, geometry):
    """Band name for each latitude under `geometry`. Ties go to the north."""
    lat = np.asarray(lat, dtype=float)
    out = np.empty(lat.shape, dtype=object)
    for name, south, north in BAND_EDGES[geometry]:
        sel = (lat >= south) & (lat <= north)
        out[sel] = name
    return out


def fit_bands(st_lat, st_value, geometry):
    """Band means from a set of stations -> {band: value}.

    A band with no station falls back to the mean of everything, which is the
    honest thing to do: it says "no information here" rather than inventing a
    contrast. This is what makes the Graven three-band geometry scoreable at
    all -- the tropics carry at most one or two stations in most months.
    """
    st_lat = np.asarray(st_lat, dtype=float)
    st_value = np.asarray(st_value, dtype=float)
    bands = band_of(st_lat, geometry)
    fallback = float(st_value.mean())
    return {
        name: (float(st_value[bands == name].mean())
               if np.any(bands == name) else fallback)
        for name, _, _ in BAND_EDGES[geometry]
    }


def delta_atm_zonal(lat, band_values, geometry="miller_band", blend_deg=0.0):
    """Zonally uniform Delta_atm(lat) [permil] for one geometry.

    lat         : latitudes [degrees], array or xr.DataArray
    band_values : {band_name: permil}, e.g. from fit_bands
    blend_deg   : half-width of a linear blend across each internal boundary.
                  0 reproduces the hard step every one of these geometries has
                  in the literature. A few degrees removes a discontinuity that
                  a transport model would otherwise see in the forcing -- the
                  straight lines visible at 10N and 40S in Miller et al.'s
                  Figure 1b are exactly this artefact.
    """
    edges = BAND_EDGES[geometry]
    lat_arr = np.asarray(lat, dtype=float)

    field = np.full(lat_arr.shape, np.nan)
    for name, south, north in edges:
        field = np.where((lat_arr >= south) & (lat_arr <= north),
                         band_values[name], field)

    if blend_deg > 0:
        # Blend across each internal boundary, south band -> north band.
        for i in range(len(edges) - 1):
            lo_name, hi_name = edges[i][0], edges[i + 1][0]
            boundary = edges[i][2]
            w = np.clip((lat_arr - (boundary - blend_deg)) / (2.0 * blend_deg), 0.0, 1.0)
            near = np.abs(lat_arr - boundary) <= blend_deg
            blended = band_values[lo_name] + w * (band_values[hi_name] - band_values[lo_name])
            field = np.where(near, blended, field)

    offset = BAND_OFFSETS.get(geometry)
    if offset is not None:
        in_band = (lat_arr <= offset["band_north"]) & (lat_arr >= offset["band_south"])
        field = field + np.where(in_band, offset["band_offset"], 0.0)

    if isinstance(lat, xr.DataArray):
        return xr.DataArray(field, coords=lat.coords, dims=lat.dims)
    return field


def area_weighted_mean(lat, field):
    """cos(lat)-weighted mean of a zonally uniform field [permil]."""
    w = np.cos(np.radians(np.asarray(lat, dtype=float)))
    return float(np.sum(w * np.asarray(field, dtype=float)) / np.sum(w))


def relevel(lat, field, global_mean):
    """Shift a zonal profile so its area-weighted mean is `global_mean`.

    This is the "shape from one source, level from another" operation. It
    exists because Hua et al. (2021) matches this repo's station network on
    the interhemispheric GRADIENT (-5 to -9 permil over 2011-2013 against our
    -6.2) while sitting ~6-8 permil above Graven et al. (2017) and Miller et
    al. (2025) on absolute LEVEL -- and Delta_atm enters the disequilibrium
    fluxes as Delta_source - Delta_atm, so level and shape have entirely
    different consequences and must be chosen separately.
    """
    return field - area_weighted_mean(lat, field) + float(global_mean)
