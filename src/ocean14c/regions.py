"""Spatial region masks for the two-population regression (CLAUDE.md Sec 3.2).

"cold/deep" population = Southern Ocean + northern North Pacific.
"warm" population = everywhere else. Drake Passage is excluded from both
when refitting the regression from scratch (not needed for v1, which uses
the published coefficients directly, but kept for reproducing Fig. S2/S5).
"""

import numpy as np
import xarray as xr

NORTH_PACIFIC_LON_SPLIT = -149.5  # degrees east; boundary between the two sub-bounds
NORTH_PACIFIC_WEST_LAT = 32.0  # deg N, west of the split
NORTH_PACIFIC_EAST_LAT = 29.0  # deg N, east of the split

# Rough North Pacific basin bounds so the lat/lon threshold above doesn't
# leak into the North Atlantic at the same latitudes. Not given explicitly
# in CLAUDE.md/the SI -- a necessary implementation choice, documented here.
NORTH_PACIFIC_BASIN_LON_MIN = 120.0  # deg E
NORTH_PACIFIC_BASIN_LON_MAX = -100.0  # deg E (i.e. -180 through here, or >=120)

# Drake Passage box, per SI Fig. S2 caption ("black crosses").
DRAKE_PASSAGE_LAT = (-65.0, -55.0)
DRAKE_PASSAGE_LON = (-70.0, -55.0)


def southern_ocean_mask(delta_1995: xr.DataArray, threshold: float = 20.0) -> xr.DataArray:
    """Cells south of the 20 permil isoline (dynamic, per-longitude boundary).

    For each longitude column, cells are Southern Ocean while contiguously
    connected to the South Pole (lat=-89.5) and below `threshold`; the mask
    stops at the first crossing moving north, even if the field dips back
    below threshold again further north (CLAUDE.md Sec 1/3.2: "dynamic
    boundary...the 20 permil isoline"). Missing data (land, ice-covered, no
    bottle samples) doesn't break the connection -- only an actual value at
    or above `threshold` does.
    """
    lat_sorted = delta_1995.sortby("lat")

    def _flood_from_south(column: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(column, dtype=bool)
        still_connected = True
        for i, v in enumerate(column):
            if np.isnan(v):
                mask[i] = still_connected
                continue
            still_connected = still_connected and (v <= threshold)
            mask[i] = still_connected
        return mask

    mask = xr.apply_ufunc(
        _flood_from_south,
        lat_sorted,
        input_core_dims=[["lat"]],
        output_core_dims=[["lat"]],
        vectorize=True,
    )
    return mask.sortby("lat", ascending=(delta_1995["lat"].values[0] < delta_1995["lat"].values[-1]))


def northern_north_pacific_mask(field: xr.DataArray) -> xr.DataArray:
    """Northern North Pacific: 32N west of 149.5W, 29N east of it, within the
    North Pacific basin longitude range."""
    lat, lon = field["lat"], field["lon"]

    in_basin = (lon >= NORTH_PACIFIC_BASIN_LON_MIN) | (lon <= NORTH_PACIFIC_BASIN_LON_MAX)
    west_of_split = lon <= NORTH_PACIFIC_LON_SPLIT
    north_enough = xr.where(
        west_of_split, lat > NORTH_PACIFIC_WEST_LAT, lat > NORTH_PACIFIC_EAST_LAT
    )
    return (in_basin & north_enough).broadcast_like(field).astype(bool)


def drake_passage_mask(field: xr.DataArray) -> xr.DataArray:
    """Drake Passage box, excluded from regression fitting (too much
    cross-frontal scatter, CLAUDE.md Sec 3.2)."""
    lat, lon = field["lat"], field["lon"]
    mask = (
        (lat >= DRAKE_PASSAGE_LAT[0])
        & (lat <= DRAKE_PASSAGE_LAT[1])
        & (lon >= DRAKE_PASSAGE_LON[0])
        & (lon <= DRAKE_PASSAGE_LON[1])
    )
    return mask.broadcast_like(field).astype(bool)


def is_cold_population(delta_1995: xr.DataArray, threshold: float = 20.0) -> xr.DataArray:
    """cold/deep population = Southern Ocean union northern North Pacific."""
    return southern_ocean_mask(delta_1995, threshold=threshold) | northern_north_pacific_mask(
        delta_1995
    )
