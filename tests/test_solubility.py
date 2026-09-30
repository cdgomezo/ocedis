import numpy as np
import pytest
import xarray as xr

from ocean14c.solubility import co2_solubility


def test_co2_solubility_20c_s35_matches_wanninkhof1992_anchor():
    # Wanninkhof (1992) text: "a CO2 solubility of 0.0324 mol/L atm at 20C"
    # (used in their own bomb-14C gamma calibration).
    assert co2_solubility(20.0, salinity=35.0) == pytest.approx(0.0324, abs=2e-4)


def test_co2_solubility_decreases_with_temperature():
    # CO2 is less soluble in warmer water.
    assert co2_solubility(5.0, salinity=35.0) > co2_solubility(30.0, salinity=35.0)


def test_co2_solubility_positive():
    assert co2_solubility(0.0, salinity=35.0) > 0


def test_co2_solubility_preserves_xarray_dims():
    # Same failure mode as foa_flux: np.asarray() on the input would strip
    # dim/coordinate metadata, breaking dim-name-based broadcasting later
    # in the pipeline when this result is combined with other fields.
    sst = xr.DataArray(
        np.full((2, 3), 15.0), dims=("month", "lat"), coords={"month": [1, 2], "lat": [0.0, 1.0, 2.0]}
    )
    result = co2_solubility(sst)
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("month", "lat")
