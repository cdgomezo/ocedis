"""Writer for the gridded flux product (ocean14c.output).

Checks the file and variable naming convention, and the NaN-to-zero land
handling that keeps a transport model from ingesting NaN.
"""

import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr

from ocean14c.output import write_year


def _flux(nlat=4, nlon=8, fill=-2.5e-2):
    return xr.DataArray(
        np.full((12, nlat, nlon), fill, dtype=float),
        dims=("month", "lat", "lon"),
        coords={
            "month": np.arange(1, 13),
            "lat": np.linspace(-89.5, -86.5, nlat),
            "lon": np.linspace(-179.5, -172.5, nlon),
        },
    )


def test_filename_follows_convention(tmp_path):
    path = write_year(2006, _flux(), tmp_path)
    assert (tmp_path / "flux_c14.ocedis.2006.nc").exists()
    assert path.endswith("flux_c14.ocedis.2006.nc")


def test_variable_name_dims_and_units(tmp_path):
    write_year(2006, _flux(), tmp_path)
    with nc.Dataset(tmp_path / "flux_c14.ocedis.2006.nc") as ds:
        assert "c14flux" in ds.variables
        v = ds.variables["c14flux"]
        assert v.dimensions == ("time", "lat", "lon")
        assert v.units == "umol m-2 s-1"
        assert "lev" not in ds.dimensions


def test_values_preserved(tmp_path):
    flux = _flux()
    flux.values[:] = np.random.default_rng(0).normal(-2e-2, 5e-3, flux.shape)
    write_year(2006, flux, tmp_path)
    with nc.Dataset(tmp_path / "flux_c14.ocedis.2006.nc") as ds:
        assert np.allclose(ds.variables["c14flux"][:], flux.values, rtol=1e-6)


def test_land_nans_written_as_zero(tmp_path):
    """TM5 emission fields must not contain NaN -- one would poison the tracer."""
    flux = _flux()
    flux.values[:, 0, 0] = np.nan
    write_year(2006, flux, tmp_path)
    with nc.Dataset(tmp_path / "flux_c14.ocedis.2006.nc") as ds:
        stored = ds.variables["c14flux"][:].data
        assert not np.isnan(stored).any()
        assert stored[0, 0, 0] == 0.0


def test_rejects_wrong_number_of_months(tmp_path):
    flux = _flux().isel(month=slice(0, 6))
    with pytest.raises(ValueError, match="12 months"):
        write_year(2006, flux, tmp_path)


def test_time_axis_is_month_starts(tmp_path):
    write_year(2006, _flux(), tmp_path)
    with nc.Dataset(tmp_path / "flux_c14.ocedis.2006.nc") as ds:
        t = ds.variables["time"]
        assert t.units == "days since 2006-01-01 00:00:00"
        assert list(t[:]) == [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


def test_time_axis_handles_leap_year(tmp_path):
    """1 March is day 60 in a leap year, not 59. The old hardcoded table got
    this wrong for every leap year; the offsets are now computed."""
    write_year(2008, _flux(), tmp_path)
    with nc.Dataset(tmp_path / "flux_c14.ocedis.2008.nc") as ds:
        assert list(ds.variables["time"][:]) == [
            0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335
        ]


def test_term_selects_the_filename_and_defaults_to_the_reconstruction(tmp_path):
    """The Table 1 variant sits beside the reconstruction, never on top of it."""
    write_year(2006, _flux(), tmp_path)
    write_year(2006, _flux(), tmp_path, term="ocedis_table1")
    assert (tmp_path / "flux_c14.ocedis.2006.nc").exists()
    assert (tmp_path / "flux_c14.ocedis_table1.2006.nc").exists()


def test_scaling_provenance_is_recorded_in_the_file(tmp_path):
    """A flux file must say which variant it holds -- the two are the same
    field to within a scalar, so nothing in the data itself distinguishes
    them."""
    write_year(2006, _flux(), tmp_path)
    with nc.Dataset(tmp_path / "flux_c14.ocedis.2006.nc") as ds:
        assert ds.term == "ocedis"
        assert "unscaled" in ds.scaling

    write_year(2006, _flux(), tmp_path, term="ocedis_table1",
               scaling="2006: x1.1169  [Table 1 target -2278.8 PgC permil/yr]")
    with nc.Dataset(tmp_path / "flux_c14.ocedis_table1.2006.nc") as ds:
        assert ds.term == "ocedis_table1"
        assert "1.1169" in ds.scaling
