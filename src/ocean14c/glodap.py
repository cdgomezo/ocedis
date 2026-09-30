"""Load the GLODAP gridded surface ocean Delta14C-DIC climatology (Key et al.
2004), treated as representative of 1995 conditions (CLAUDE.md Sec 3.1).

File format (data/raw/GLODAP/): C14.data.txt has one row per (depth, lon)
combination -- row index = depth_idx * n_lon + lon_idx -- with one
comma-separated value per latitude. This ordering was determined empirically
by spot-checking gridded values against the raw station list (C14Sta.0.txt):
the correct ordering gives a median absolute error of ~11 permil against
individual bottle samples, the wrong (lon-major) ordering ~64 permil.
"""

from pathlib import Path

import numpy as np
import xarray as xr

FILL_VALUE = -999.0
SURFACE_MAX_DEPTH_M = 50.0  # Miller et al. 2025's own definition of "surface"


def _read_csv_floats(path: Path):
    with open(path) as f:
        return np.array([float(x) for x in f.read().strip().split(",")])


def _read_grid_rows(path: Path):
    with open(path) as f:
        return np.array(
            [[float(x) for x in line.strip().split(",")] for line in f if line.strip()]
        )


def load_c14_grid(raw_dir) -> xr.DataArray:
    """Full 3D (depth, lon, lat) Delta14C-DIC grid, NaN where missing."""
    raw_dir = Path(raw_dir)
    lat = _read_csv_floats(raw_dir / "Lat.centers.txt")
    lon = _read_csv_floats(raw_dir / "Long.centers.txt")
    depth = _read_csv_floats(raw_dir / "Depth.centers.txt")

    rows = _read_grid_rows(raw_dir / "C14.data.txt")
    n_lon, n_depth, n_lat = len(lon), len(depth), len(lat)
    data = rows.reshape(n_depth, n_lon, n_lat)
    data = np.where(data == FILL_VALUE, np.nan, data)

    return xr.DataArray(
        data,
        dims=("depth", "lon", "lat"),
        coords={"depth": depth, "lon": lon, "lat": lat},
        name="delta14c_dic",
        attrs={"units": "permil"},
    )


def surface_delta_1995(raw_dir, max_depth_m: float = SURFACE_MAX_DEPTH_M) -> xr.DataArray:
    """Surface (<50 m) Delta14C-DIC field, averaged over depth, for (lon, lat)."""
    grid = load_c14_grid(raw_dir)
    surface_levels = grid.sel(depth=grid["depth"] <= max_depth_m)
    return surface_levels.mean(dim="depth", skipna=True)


def load_stations(raw_dir):
    """Raw station observations from C14Sta.0.txt as (lon, lat, c14) tuples."""
    raw_dir = Path(raw_dir)
    stations = []
    with open(raw_dir / "C14Sta.0.txt") as f:
        next(f)  # header: row.names,station,longitude,latitude,c14
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            try:
                lon, lat, c14 = float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            stations.append((lon, lat, c14))
    return stations
