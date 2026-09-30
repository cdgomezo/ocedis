import numpy as np
import xarray as xr

from ocean14c.surface_ocean import project_delta_oce


def _field(values, lats, lons):
    return xr.DataArray(
        np.array(values, dtype=float),
        dims=("lon", "lat"),
        coords={"lon": np.array(lons, dtype=float), "lat": np.array(lats, dtype=float)},
    )


def test_project_delta_oce_at_1995_returns_unchanged_field():
    delta_1995 = _field([[-100.0, 50.0]], [-89.5, 33.0], [0.0])
    projected = project_delta_oce(delta_1995, year=1995.0)
    xr.testing.assert_allclose(projected, delta_1995)


def test_project_delta_oce_applies_cold_rate_in_southern_ocean():
    delta_1995 = _field([[-100.0]], [-89.5], [0.0])
    projected = project_delta_oce(delta_1995, year=1996.0)
    m_cold, b_cold = -0.0200, -3.0851
    expected = -100.0 + (m_cold * -100.0 + b_cold) * 1.0
    assert projected.sel(lon=0.0, lat=-89.5).item() == expected


def test_project_delta_oce_applies_warm_rate_elsewhere():
    delta_1995 = _field([[50.0]], [0.0], [0.0])
    projected = project_delta_oce(delta_1995, year=1996.0)
    m_warm, b_warm = -0.0549, 2.9832
    expected = 50.0 + (m_warm * 50.0 + b_warm) * 1.0
    assert projected.sel(lon=0.0, lat=0.0).item() == expected
