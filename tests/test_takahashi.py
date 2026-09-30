from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ocean14c.takahashi import (
    load_pco2_climatology,
    load_sst_climatology,
    project_pco2_ocean,
    regrid_nearest,
)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "Takahashi2009"
requires_data = pytest.mark.skipif(
    not (RAW_DIR / "month_flux_2006c.txt").exists(), reason="Takahashi raw data not present"
)


@requires_data
def test_load_pco2_climatology_has_expected_dims():
    clim = load_pco2_climatology(RAW_DIR / "month_flux_2006c.txt")
    assert set(clim.dims) == {"month", "lat", "lon"}
    assert clim.sizes["month"] == 12
    assert clim.sizes["lat"] == 40
    assert clim.sizes["lon"] == 72


@requires_data
def test_load_pco2_climatology_known_value():
    # First data row of month_flux_2006c.txt: LAT=0 LON=2.5 MONTH=1 PCO2_SW=378.81
    clim = load_pco2_climatology(RAW_DIR / "month_flux_2006c.txt")
    lon = 2.5 if 2.5 in clim["lon"].values else 2.5 - 360
    val = clim.sel(month=1, lat=0, lon=clim["lon"].values[np.argmin(np.abs(clim["lon"].values - lon))])
    assert val.item() == pytest.approx(378.81)


@requires_data
def test_load_sst_climatology_has_expected_dims():
    sst = load_sst_climatology(RAW_DIR / "month_flux_2006c.txt")
    assert set(sst.dims) == {"month", "lat", "lon"}
    assert sst.sizes["month"] == 12


@requires_data
def test_load_sst_climatology_known_value():
    # First data row of month_flux_2006c.txt: LAT=0 LON=2.5 MONTH=1 SST=28.080
    # (not MOD_SST=26.49, a different column earlier in the same row --
    # verified against the file's header line before fixing this test).
    sst = load_sst_climatology(RAW_DIR / "month_flux_2006c.txt")
    lon = sst["lon"].values[np.argmin(np.abs(sst["lon"].values - 2.5))]
    val = sst.sel(month=1, lat=0, lon=lon)
    assert val.item() == pytest.approx(28.080)


def test_regrid_nearest_picks_closest_native_cell():
    native = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=("lat", "lon"),
        coords={"lat": [0.0, 10.0], "lon": [0.0, 10.0]},
    )
    target_lat = np.array([1.0, 9.0])
    target_lon = np.array([1.0, 9.0])
    regridded = regrid_nearest(native, target_lat, target_lon)
    assert regridded.sel(lat=1.0, lon=1.0).item() == 1.0
    assert regridded.sel(lat=9.0, lon=9.0).item() == 4.0


def test_regrid_nearest_skips_nan_native_cells():
    # The geometrically nearest native cell to (9, 9) is the NaN one
    # (10, 10) -- a naive nearest-neighbor regrid would propagate that NaN,
    # silently dropping real ocean cells wherever the closest native grid
    # point happens to be land (this bit us for real: 62% of GLODAP's ocean
    # area was going NaN after regridding the coarse Takahashi grid before
    # this fix). The nearest *valid* cell to (9, 9) is (10, 0) = 2.0.
    native = xr.DataArray(
        [[1.0, 2.0], [3.0, np.nan]],
        dims=("lat", "lon"),
        coords={"lat": [0.0, 10.0], "lon": [0.0, 10.0]},
    )
    target_lat = np.array([9.0])
    target_lon = np.array([9.0])
    regridded = regrid_nearest(native, target_lat, target_lon)
    assert regridded.sel(lat=9.0, lon=9.0).item() == 2.0


def test_regrid_nearest_skips_nan_across_a_month_dimension():
    native = xr.DataArray(
        [[[1.0, 2.0], [3.0, np.nan]], [[5.0, 6.0], [7.0, np.nan]]],
        dims=("month", "lat", "lon"),
        coords={"month": [1, 2], "lat": [0.0, 10.0], "lon": [0.0, 10.0]},
    )
    regridded = regrid_nearest(native, np.array([9.0]), np.array([9.0]))
    assert regridded.sel(month=1, lat=9.0, lon=9.0).item() == 2.0
    assert regridded.sel(month=2, lat=9.0, lon=9.0).item() == 6.0


def test_project_pco2_ocean_applies_drift_from_year_2000():
    field = xr.DataArray([[380.0]], dims=("month", "cell"), coords={"month": [1], "cell": [0]})
    projected = project_pco2_ocean(field, year=2006, drift_rate=1.5)
    assert projected.item() == pytest.approx(380.0 + 1.5 * 6)


def test_project_pco2_ocean_at_year_2000_is_unchanged():
    field = xr.DataArray([[380.0]], dims=("month", "cell"), coords={"month": [1], "cell": [0]})
    projected = project_pco2_ocean(field, year=2000, drift_rate=1.5)
    assert projected.item() == pytest.approx(380.0)
