"""Takahashi et al. (2009) surface ocean pCO2 climatology (CLAUDE.md Sec 4.5).

Monthly climatology on a 4deg lat x 5deg lon grid, referenced to year 2000
and already normalized at +1.5 uatm/yr (per the file's own header text,
matching CLAUDE.md's drift rate). We regrid to the target 1x1 grid with
nearest-neighbor interpolation and re-apply the same drift for other years.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

HEADER_LINE = 108  # 0-indexed rows to skip; header itself is row 109 (1-indexed)
REFERENCE_YEAR = 2000.0
DEFAULT_DRIFT_RATE = 1.5  # uatm/yr


def _load_column(path, column: str, name: str) -> xr.DataArray:
    df = pd.read_csv(path, sep=r"\s+", skiprows=HEADER_LINE)
    df = df[["MONTH", "LAT", "LON", column]].rename(
        columns={"MONTH": "month", "LAT": "lat", "LON": "lon", column: name}
    )
    indexed = df.set_index(["month", "lat", "lon"])[name]
    da = xr.DataArray.from_series(indexed)
    return da.rename(name)


def load_pco2_climatology(path) -> xr.DataArray:
    """Monthly PCO2_SW climatology (year-2000 reference), dims (month, lat, lon)."""
    return _load_column(path, "PCO2_SW", "pco2_sw")


def load_sst_climatology(path) -> xr.DataArray:
    """Monthly climatological SST [deg C], dims (month, lat, lon). No secular
    drift is applied -- this is a fixed 1981-2006 seasonal climatology (per
    the file's own header text), unlike PCO2_SW which does drift."""
    return _load_column(path, "SST", "sst")


def regrid_nearest(field: xr.DataArray, target_lat, target_lon) -> xr.DataArray:
    """Nearest-*valid*-neighbor regrid of `field`'s lat/lon dims onto target
    coordinates.

    A naive nearest-neighbor regrid (e.g. DataArray.interp(method="nearest"))
    propagates NaN whenever the geometrically nearest native cell happens to
    be land -- which matters a lot for the coarse-to-fine regrid this
    project needs (Takahashi's native 4x5deg grid onto the target 1x1deg
    grid): naively, 62% of GLODAP's ocean-cell area went NaN after
    regridding before this fix, because many 1x1deg ocean cells' nearest
    4x5deg-grid center is a native land/no-data cell. This implementation
    excludes NaN native cells from the nearest-neighbor search entirely, so
    every target ocean cell gets the nearest *valid* native value.
    """
    target_lat = np.asarray(target_lat, dtype=float)
    target_lon = np.asarray(target_lon, dtype=float)
    field = field.transpose(..., "lat", "lon")

    native_lat = field["lat"].values
    native_lon = field["lon"].values
    lat_grid, lon_grid = np.meshgrid(native_lat, native_lon, indexing="ij")

    target_lat_grid, target_lon_grid = np.meshgrid(target_lat, target_lon, indexing="ij")
    target_points = np.column_stack([target_lat_grid.ravel(), target_lon_grid.ravel()])

    def _regrid_slice(values_2d):
        valid = ~np.isnan(values_2d)
        valid_points = np.column_stack([lat_grid[valid], lon_grid[valid]])
        tree = cKDTree(valid_points)
        _, idx = tree.query(target_points)
        return values_2d[valid][idx].reshape(len(target_lat), len(target_lon))

    regridded = xr.apply_ufunc(
        _regrid_slice,
        field,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        exclude_dims={"lat", "lon"},
        vectorize=True,
    )
    return regridded.assign_coords(lat=target_lat, lon=target_lon)


def project_pco2_ocean(
    climatology: xr.DataArray, year: float, drift_rate: float = DEFAULT_DRIFT_RATE
) -> xr.DataArray:
    """Drift the year-2000-reference climatology forward/backward to `year`."""
    return climatology + drift_rate * (year - REFERENCE_YEAR)
