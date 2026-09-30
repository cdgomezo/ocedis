from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ocean14c.atmosphere_forcing import (
    _time_coord_name,
    compute_monthly_u2,
    process_year,
)


def _write_raw_fixture(path: Path, time_dim_name: str):
    # Six-hourly u10/v10 for January (4 steps/day * 31 days) at one grid cell,
    # with a known, deliberately non-constant wind speed so that
    # mean(u10^2+v10^2) != mean(u10)^2 + mean(v10)^2 -- this is the whole
    # point of using the second moment, not the square of the mean.
    n_steps = 4 * 31
    times = np.array(
        ["2006-01-01"], dtype="datetime64[D]"
    ) + np.arange(n_steps) * np.timedelta64(6, "h")
    times = times.astype("datetime64[ns]")

    rng = np.random.default_rng(42)
    u10 = rng.normal(loc=5.0, scale=3.0, size=n_steps)
    v10 = rng.normal(loc=-2.0, scale=1.5, size=n_steps)

    ds = xr.Dataset(
        {
            "u10": ((time_dim_name, "latitude", "longitude"), u10[:, None, None]),
            "v10": ((time_dim_name, "latitude", "longitude"), v10[:, None, None]),
        },
        coords={time_dim_name: times, "latitude": [10.0], "longitude": [20.0]},
    )
    ds.to_netcdf(path)
    return u10, v10


def test_time_coord_name_finds_valid_time():
    ds = xr.Dataset(coords={"valid_time": [0], "latitude": [0], "longitude": [0]})
    assert _time_coord_name(ds) == "valid_time"


def test_time_coord_name_finds_time():
    ds = xr.Dataset(coords={"time": [0], "latitude": [0], "longitude": [0]})
    assert _time_coord_name(ds) == "time"


def test_time_coord_name_raises_when_neither_present():
    ds = xr.Dataset(coords={"latitude": [0], "longitude": [0]})
    with pytest.raises(KeyError):
        _time_coord_name(ds)


@pytest.mark.parametrize("time_dim_name", ["time", "valid_time"])
def test_compute_monthly_u2_is_second_moment_not_square_of_mean(tmp_path, time_dim_name):
    raw_path = tmp_path / "raw.nc"
    u10, v10 = _write_raw_fixture(raw_path, time_dim_name)

    monthly = compute_monthly_u2(raw_path)

    expected_second_moment = np.mean(u10**2 + v10**2)
    wrong_square_of_mean = np.mean(u10) ** 2 + np.mean(v10) ** 2
    assert float(monthly.isel(**{monthly.dims[0]: 0, "latitude": 0, "longitude": 0})) == pytest.approx(
        expected_second_moment
    )
    assert expected_second_moment != pytest.approx(wrong_square_of_mean)


def test_process_year_downloads_processes_and_deletes_raw_file(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()

    calls = []

    def fake_download_year(year, raw_path):
        calls.append(year)
        _write_raw_fixture(raw_path, "valid_time")

    monkeypatch.setattr("ocean14c.atmosphere_forcing.download_year", fake_download_year)

    output_path = process_year(2006, output_dir=output_dir, tmp_dir=tmp_dir)

    assert calls == [2006]
    assert output_path.exists()
    assert list(tmp_dir.iterdir()) == []  # raw file deleted after processing


def test_process_year_is_resumable_and_skips_existing_output(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing = output_dir / "u2_2006.nc"
    existing.write_text("already processed")

    calls = []
    monkeypatch.setattr(
        "ocean14c.atmosphere_forcing.download_year", lambda year, raw_path: calls.append(year)
    )

    result = process_year(2006, output_dir=output_dir, tmp_dir=tmp_path)

    assert calls == []  # never re-downloaded
    assert result == existing
    assert existing.read_text() == "already processed"
