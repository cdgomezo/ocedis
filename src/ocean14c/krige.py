"""Spatial interpolation of station Delta14C onto a grid, one month at a time.

This is the spatial half of the gridded Delta_atm construction; ocean14c.curvefit
handles time. Because each station is already fitted to a continuous curve, every
month is an independent problem with at most ~20 points, so the linear systems are
tiny and solved directly -- no external kriging dependency.

Method: universal kriging (kriging with external drift). Delta14C is dominated by
a large secular trend and a smooth interhemispheric gradient, which ordinary
kriging handles badly; universal kriging carries the trend in a drift basis and
krige only what is left, and -- unlike fitting the drift separately -- returns a
prediction variance that accounts for uncertainty in the drift itself.

Anisotropy is deliberate. Zonal mixing in the atmosphere takes weeks, meridional
mixing months, so the zonal correlation length should greatly exceed the
meridional one. Rather than assert that, both lengths are fitted by
leave-one-out cross-validation (fit_length_scales), which lets the data decide
how much longitudinal structure the network can actually resolve.

The single entry point a future optimal-interpolation scheme would replace is
`krige_month`: {station lat/lon/value/error} -> {gridded field, uncertainty}.
"""

import numpy as np

EARTH_RADIUS_KM = 6371.0

# --------------------------------------------------------------------------
# Defaults, calibrated by leave-one-station-out cross-validation over
# 1990-2023 background stations (scripts/validate_delta_atm_grid.py).
#
# WHAT THE CROSS-VALIDATION FOUND -- read this before tuning anything:
#
#   global mean (no spatial structure)   3.570 permil
#   latitude-band mean (+-20 deg)        3.394
#   drift only, sin(lat) degree 1        2.921
#   drift only, sin(lat) degree 2        3.077
#   drift only, sin(lat) degree 3        3.746   (overfits)
#   universal kriging, degree 1, best    2.918
#
# The station network supports a LINEAR interhemispheric gradient and
# essentially nothing more. Residual kriging improves LOO RMSE by 0.1%
# (2.921 -> 2.918), and the RMSE surface is flat to within 1% across a 200x
# range of correlation lengths -- i.e. neither longitudinal structure nor
# meridional curvature is resolvable from these stations. Degree 2 is
# measurably worse than degree 1, so the default drift is degree 1.
#
# The kriging machinery is kept regardless, for two reasons: it yields the
# spatially varying UNCERTAINTY field (which a bare least-squares fit does
# not), and it is the natural slot for the planned optimal-interpolation
# scheme, which will have transport information the current network lacks.
# --------------------------------------------------------------------------
DEFAULT_L_LAT_KM = 1000.0
DEFAULT_L_LON_KM = 8000.0
DEFAULT_SILL = 1.0        # permil^2, variance of the residual field
DEFAULT_DRIFT_DEGREE = 1  # linear in sin(lat)

# Observation error used in the kriging system. This is a SPATIAL
# REPRESENTATIVENESS error, not the scatter of a station about its own fitted
# curve. Weighting by curve scatter was tried and is actively harmful: it
# over-weights stations whose records happen to be smooth (SMO, MLO, SPO at
# ~1 permil) over stations that are more representative of the large-scale
# field, and pushed LOO RMSE from 2.921 to 3.105.
DEFAULT_OBS_ERR = 2.0     # permil, 1-sigma
REGIONAL_OBS_ERR = 6.0    # permil, for sites with known local fossil influence
DEFAULT_NUGGET = 0.25     # permil^2, numerical floor on the diagonal


def drift_basis(lat, degree=2):
    """Drift basis evaluated at `lat` [degrees]: [1, sin(lat), sin^2(lat), ...].

    sin(lat) rather than lat because the interhemispheric gradient is smooth in
    sin(lat) and it keeps the basis well conditioned over the full sphere.
    """
    s = np.sin(np.radians(np.asarray(lat, dtype=float)))
    return np.column_stack([s**k for k in range(degree + 1)])


def _anisotropic_distance(lat1, lon1, lat2, lon2, l_lat_km, l_lon_km):
    """Dimensionless separation under an anisotropic metric.

    Meridional and zonal separations are converted to km and divided by their
    own correlation lengths, so the result is directly usable in an isotropic
    covariance function. Longitude differences are wrapped to +-180 and scaled
    by cos(mean latitude) so a degree of longitude shrinks toward the poles.
    """
    lat1 = np.asarray(lat1, float)[:, None]
    lon1 = np.asarray(lon1, float)[:, None]
    lat2 = np.asarray(lat2, float)[None, :]
    lon2 = np.asarray(lon2, float)[None, :]

    dlat_km = np.radians(lat2 - lat1) * EARTH_RADIUS_KM

    dlon = (lon2 - lon1 + 180.0) % 360.0 - 180.0
    cos_mid = np.cos(np.radians(0.5 * (lat1 + lat2)))
    dlon_km = np.radians(dlon) * EARTH_RADIUS_KM * cos_mid

    return np.sqrt((dlat_km / l_lat_km) ** 2 + (dlon_km / l_lon_km) ** 2)


def covariance(d, sill=DEFAULT_SILL):
    """Exponential covariance for dimensionless separation `d`."""
    return sill * np.exp(-d)


def krige_month(
    st_lat, st_lon, st_value, st_err,
    grid_lat, grid_lon,
    l_lat_km=DEFAULT_L_LAT_KM,
    l_lon_km=DEFAULT_L_LON_KM,
    sill=DEFAULT_SILL,
    drift_degree=DEFAULT_DRIFT_DEGREE,
):
    """Universal kriging of one month onto a lat/lon grid.

    st_*      : station latitude, longitude, value [permil], 1-sigma error [permil]
    grid_lat  : (nlat,) target latitudes
    grid_lon  : (nlon,) target longitudes

    Returns (field, sd), both (nlat, nlon): the predicted Delta14C and the
    kriging standard deviation.
    """
    st_lat = np.asarray(st_lat, float)
    st_lon = np.asarray(st_lon, float)
    z = np.asarray(st_value, float)
    n = len(z)

    if n == 0:
        shape = (len(grid_lat), len(grid_lon))
        return np.full(shape, np.nan), np.full(shape, np.nan)

    F = drift_basis(st_lat, drift_degree)
    p = F.shape[1]
    # With fewer stations than drift terms the system is not identifiable;
    # fall back to as many terms as can be supported.
    while p >= n and p > 1:
        F = F[:, : p - 1]
        p = F.shape[1]

    d_ss = _anisotropic_distance(st_lat, st_lon, st_lat, st_lon, l_lat_km, l_lon_km)
    K = covariance(d_ss, sill)
    K[np.diag_indices(n)] += np.maximum(np.asarray(st_err, float) ** 2, DEFAULT_NUGGET)

    # Bordered universal-kriging system: [[K, F], [F', 0]]
    A = np.zeros((n + p, n + p))
    A[:n, :n] = K
    A[:n, n:] = F
    A[n:, :n] = F.T
    A += np.eye(n + p) * 1e-10          # guard against exact singularity

    glat, glon = np.meshgrid(grid_lat, grid_lon, indexing="ij")
    tgt_lat, tgt_lon = glat.ravel(), glon.ravel()

    d_st = _anisotropic_distance(st_lat, st_lon, tgt_lat, tgt_lon, l_lat_km, l_lon_km)
    k = covariance(d_st, sill)                       # (n, ntarget)
    f = drift_basis(tgt_lat, drift_degree)[:, :p].T  # (p, ntarget)

    rhs = np.vstack([k, f])
    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, rhs, rcond=None)[0]

    weights = sol[:n, :]
    field = weights.T @ z

    # var = sill - (w'k + mu'f); clipped at 0 for numerical safety
    var = sill - np.einsum("ij,ij->j", sol, rhs)
    sd = np.sqrt(np.clip(var, 0.0, None))

    shape = (len(grid_lat), len(grid_lon))
    return field.reshape(shape), sd.reshape(shape)


def krige_points(
    st_lat, st_lon, st_value, st_err,
    tgt_lat, tgt_lon,
    **kwargs,
):
    """Same as krige_month but for scattered target points rather than a grid.

    Used by the leave-one-out cross-validation, where the target is a single
    withheld station.
    """
    tgt_lat = np.atleast_1d(tgt_lat)
    tgt_lon = np.atleast_1d(tgt_lon)
    out_v, out_s = [], []
    for la, lo in zip(tgt_lat, tgt_lon):
        f, s = krige_month(
            st_lat, st_lon, st_value, st_err,
            np.array([la]), np.array([lo]),
            **kwargs,
        )
        out_v.append(f[0, 0])
        out_s.append(s[0, 0])
    return np.array(out_v), np.array(out_s)


def loo_rmse(months_data, l_lat_km, l_lon_km, sill=DEFAULT_SILL,
             drift_degree=DEFAULT_DRIFT_DEGREE):
    """Leave-one-station-out RMSE over many months, for a given length pair.

    months_data: iterable of (lat, lon, value, err) arrays, one tuple per month.
    """
    sq, cnt = 0.0, 0
    for lat, lon, val, err in months_data:
        n = len(val)
        if n < 4:
            continue
        for i in range(n):
            keep = np.arange(n) != i
            pred, _ = krige_points(
                lat[keep], lon[keep], val[keep], err[keep],
                lat[i], lon[i],
                l_lat_km=l_lat_km, l_lon_km=l_lon_km,
                sill=sill, drift_degree=drift_degree,
            )
            if np.isfinite(pred[0]):
                sq += (pred[0] - val[i]) ** 2
                cnt += 1
    return np.sqrt(sq / cnt) if cnt else np.nan


def fit_length_scales(months_data, l_lat_grid, l_lon_grid, **kwargs):
    """Pick (l_lat_km, l_lon_km) minimising leave-one-station-out RMSE.

    Optimising cross-validation error directly, rather than fitting a
    variogram, keeps the criterion identical to how the product is judged --
    and makes "longitude adds nothing" a visible outcome (the RMSE surface is
    then flat in l_lon).

    Returns (best_l_lat, best_l_lon, rmse_table) with rmse_table shaped
    (len(l_lat_grid), len(l_lon_grid)).
    """
    table = np.full((len(l_lat_grid), len(l_lon_grid)), np.nan)
    for i, ll in enumerate(l_lat_grid):
        for j, lo in enumerate(l_lon_grid):
            table[i, j] = loo_rmse(months_data, ll, lo, **kwargs)
    if np.all(np.isnan(table)):
        return DEFAULT_L_LAT_KM, DEFAULT_L_LON_KM, table
    i, j = np.unravel_index(np.nanargmin(table), table.shape)
    return float(l_lat_grid[i]), float(l_lon_grid[j]), table
