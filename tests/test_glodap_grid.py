from pathlib import Path

import numpy as np
import pytest

from ocean14c.glodap import load_c14_grid, load_stations, surface_delta_1995

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "GLODAP"
requires_data = pytest.mark.skipif(
    not RAW_DIR.exists(), reason="GLODAP raw data not present"
)


@requires_data
def test_load_c14_grid_has_expected_shape_and_axes():
    grid = load_c14_grid(RAW_DIR)
    assert grid.sizes["depth"] == 33
    assert grid.sizes["lon"] == 360
    assert grid.sizes["lat"] == 180
    assert grid["depth"].values[0] == 0.0
    assert grid["lon"].min() == pytest.approx(-179.5)
    assert grid["lat"].max() == pytest.approx(89.5)


@requires_data
def test_load_c14_grid_treats_fill_value_as_nan():
    grid = load_c14_grid(RAW_DIR)
    assert np.isnan(grid.values).any()
    assert not np.any(grid.values == -999)


@requires_data
def test_surface_field_matches_raw_stations_closely_when_row_order_is_correct():
    # This is the empirical check that pins down the file's row-major
    # ordering (depth-major: row = depth_idx*n_lon + lon_idx). Under the
    # wrong ordering the median absolute error against raw station c14
    # values balloons from ~11 permil to ~64 permil (verified by hand
    # against data/raw/GLODAP/C14Sta.0.txt) -- this test guards against a
    # silent regression to the wrong ordering.
    surface = surface_delta_1995(RAW_DIR)
    stations = load_stations(RAW_DIR)

    errors = []
    for lon, lat, c14 in stations:
        val = surface.sel(lon=lon, lat=lat, method="nearest").item()
        if not np.isnan(val):
            errors.append(abs(val - c14))

    assert len(errors) > 800  # most of the 936 stations should hit valid cells
    assert np.median(errors) < 20.0


@requires_data
def test_surface_delta_1995_shape():
    surface = surface_delta_1995(RAW_DIR)
    assert surface.sizes["lon"] == 360
    assert surface.sizes["lat"] == 180
    assert "depth" not in surface.dims
