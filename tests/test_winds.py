from pathlib import Path

import numpy as np
import xarray as xr

from ocean14c.winds import load_u2


def _write_fixture(path: Path, time_dim_name: str = "valid_time"):
    # Mirrors ocean14c.atmosphere_forcing's actual output: a DataArray
    # computed as u10**2 + v10**2 then resampled monthly on the time dim,
    # with longitude in native ERA5 0-360 convention. CDS output has used
    # both 'valid_time' and 'time' for this axis across API versions.
    lat = np.array([-1.0, 0.0, 1.0])
    lon = np.array([0.0, 90.0, 270.0])  # 270 == -90 once converted
    time = np.array(["2006-01-01", "2006-02-01"], dtype="datetime64[ns]")
    u2 = xr.DataArray(
        np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3),
        dims=(time_dim_name, "latitude", "longitude"),
        coords={time_dim_name: time, "latitude": lat, "longitude": lon},
        name="u10",
    )
    u2.to_dataset().to_netcdf(path)


def test_load_u2_has_month_lat_lon_dims(tmp_path):
    fixture = tmp_path / "u2_2006.nc"
    _write_fixture(fixture)

    field = load_u2(fixture)

    assert set(field.dims) == {"month", "lat", "lon"}
    assert list(field["month"].values) == [1, 2]


def test_load_u2_converts_longitude_to_minus180_180(tmp_path):
    fixture = tmp_path / "u2_2006.nc"
    _write_fixture(fixture)

    field = load_u2(fixture)

    assert field["lon"].max().item() <= 180.0
    assert field["lon"].min().item() >= -180.0
    # 270E native == -90 once converted
    assert -90.0 in field["lon"].values


def test_load_u2_preserves_values(tmp_path):
    fixture = tmp_path / "u2_2006.nc"
    _write_fixture(fixture)

    field = load_u2(fixture)

    # native (valid_time=0, latitude=-1.0, longitude=0.0) == 0.0 (first flat index)
    assert field.sel(month=1, lat=-1.0, lon=0.0).item() == 0.0


def test_load_u2_handles_time_coord_named_time_instead_of_valid_time(tmp_path):
    fixture = tmp_path / "u2_2006.nc"
    _write_fixture(fixture, time_dim_name="time")

    field = load_u2(fixture)

    assert set(field.dims) == {"month", "lat", "lon"}
    assert list(field["month"].values) == [1, 2]
