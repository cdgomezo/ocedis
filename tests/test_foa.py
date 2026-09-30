import numpy as np
import pytest
import xarray as xr

from ocean14c.foa import CM_PER_HR_TO_M_PER_YR, L_TO_M3, UATM_TO_ATM, foa_flux


def test_unit_conversion_constants_are_internally_consistent():
    # 1 cm/hr -> m/yr: (1/100 m/cm) * (24*365.25 hr/yr)
    assert CM_PER_HR_TO_M_PER_YR == pytest.approx(24 * 365.25 / 100)
    assert L_TO_M3 == pytest.approx(1000.0)
    assert UATM_TO_ATM == pytest.approx(1e-6)


def test_foa_flux_composes_conversions_correctly():
    # k=1 cm/hr, s=1 mol/(L*atm), pco2=1e6 uatm (=1 atm): Foa should equal
    # exactly the k unit-conversion factor, in mol C / (m^2 * yr).
    flux = foa_flux(k_cm_per_hr=1.0, solubility_mol_per_L_atm=1.0, pco2_uatm=1e6)
    assert flux == pytest.approx(24 * 365.25 / 100 * 1000.0)


def test_foa_flux_scales_linearly_with_each_input():
    base = foa_flux(k_cm_per_hr=10.0, solubility_mol_per_L_atm=0.03, pco2_uatm=380.0)
    doubled_k = foa_flux(k_cm_per_hr=20.0, solubility_mol_per_L_atm=0.03, pco2_uatm=380.0)
    doubled_pco2 = foa_flux(k_cm_per_hr=10.0, solubility_mol_per_L_atm=0.03, pco2_uatm=760.0)
    assert doubled_k == pytest.approx(2 * base)
    assert doubled_pco2 == pytest.approx(2 * base)


def test_foa_flux_is_positive_for_positive_inputs():
    assert foa_flux(k_cm_per_hr=10.0, solubility_mol_per_L_atm=0.03, pco2_uatm=380.0) > 0


def test_foa_flux_preserves_xarray_dims_and_broadcasts_by_name_not_position():
    # k has dims (lon, lat); pco2 has dims (month, lat, lon) -- a different
    # order/shape, as genuinely happens combining a GLODAP-grid field with a
    # Takahashi/ERA5-grid field. If foa_flux silently drops to raw ndarrays
    # (e.g. via np.asarray), this mismatches positionally and either raises
    # or silently broadcasts wrong values instead of aligning by dim name.
    k = xr.DataArray(
        np.full((3, 2), 10.0), dims=("lon", "lat"), coords={"lon": [0.0, 1.0, 2.0], "lat": [0.0, 1.0]}
    )
    pco2 = xr.DataArray(
        np.full((4, 2, 3), 380.0),
        dims=("month", "lat", "lon"),
        coords={"month": [1, 2, 3, 4], "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
    )
    result = foa_flux(k, solubility_mol_per_L_atm=0.03, pco2_uatm=pco2)
    assert set(result.dims) == {"month", "lat", "lon"}
    assert result.sizes == {"month": 4, "lat": 2, "lon": 3}
