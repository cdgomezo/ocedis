import numpy as np
import pytest
import xarray as xr

from ocean14c.isoflux import compute_isoflux, flux_weighted_mean, spatial_sum


def _field(values, lats, lons):
    return xr.DataArray(
        np.array(values, dtype=float),
        dims=("lon", "lat"),
        coords={"lon": np.array(lons, dtype=float), "lat": np.array(lats, dtype=float)},
    )


def test_compute_isoflux_multiplies_foa_and_delta_odis_elementwise():
    foa = _field([[2.0, 3.0]], [0.0, 10.0], [0.0])
    delta_odis = _field([[5.0, -4.0]], [0.0, 10.0], [0.0])
    result = compute_isoflux(foa, delta_odis)
    assert result.sel(lon=0.0, lat=0.0).item() == pytest.approx(10.0)
    assert result.sel(lon=0.0, lat=10.0).item() == pytest.approx(-12.0)


def test_spatial_sum_over_all_cells():
    field = _field([[1.0, 2.0], [3.0, 4.0]], [0.0, 10.0], [0.0, 20.0])
    assert spatial_sum(field) == pytest.approx(10.0)


def test_spatial_sum_respects_mask():
    field = _field([[1.0, 2.0], [3.0, 4.0]], [0.0, 10.0], [0.0, 20.0])
    mask = _field([[True, False], [True, False]], [0.0, 10.0], [0.0, 20.0]).astype(bool)
    assert spatial_sum(field, mask=mask) == pytest.approx(4.0)  # cells (0,0)+(20,0) = 1+3


def test_spatial_sum_ignores_nan():
    field = _field([[1.0, np.nan]], [0.0, 10.0], [0.0])
    assert spatial_sum(field) == pytest.approx(1.0)


def test_flux_weighted_mean_matches_table_s2_definition():
    # Delta_odis_mean = sum(Foa*Delta_odis) / sum(Foa) -- CLAUDE.md Sec 5.
    foa = _field([[10.0, 20.0]], [0.0, 10.0], [0.0])
    delta_odis = _field([[-50.0, -100.0]], [0.0, 10.0], [0.0])
    expected = (10.0 * -50.0 + 20.0 * -100.0) / (10.0 + 20.0)
    assert flux_weighted_mean(delta_odis, foa) == pytest.approx(expected)


# --- TM5-facing C_D14C flux product -----------------------------------------

def test_to_umol_flux_matches_explicit_conversion_chain():
    """foa [mol C m-2 yr-1] x delta [permil] -> umol C_D14C m-2 s-1."""
    from ocean14c.isoflux import SECONDS_PER_YEAR, to_umol_flux

    foa = _field([[16.0]], [0.0], [0.0])          # mol C m-2 yr-1
    delta = _field([[-50.0]], [0.0], [0.0])       # permil
    expected = 16.0 / SECONDS_PER_YEAR * (-50.0 / 1000.0) * 1e6
    assert to_umol_flux(foa, delta).sel(lon=0.0, lat=0.0).item() == pytest.approx(expected)


def test_to_umol_flux_applies_the_permil_division():
    """The /1000 is real here: Delta_odis is carried in permil, and C_D14C is
    defined with a dimensionless Delta. Omitting it would inflate by 1000x."""
    from ocean14c.isoflux import SECONDS_PER_YEAR, to_umol_flux

    foa = _field([[1.0]], [0.0], [0.0])
    delta = _field([[1000.0]], [0.0], [0.0])      # 1000 permil == Delta of 1.0
    result = to_umol_flux(foa, delta).sel(lon=0.0, lat=0.0).item()
    assert result == pytest.approx(1.0 / SECONDS_PER_YEAR * 1e6)


def test_to_umol_flux_is_negative_for_negative_disequilibrium():
    """Delta_odis < 0 over essentially the whole ocean, so the flux is negative
    even though Foa (outgassing) is positive."""
    from ocean14c.isoflux import to_umol_flux

    foa = _field([[16.0]], [0.0], [0.0])
    delta = _field([[-50.0]], [0.0], [0.0])
    assert to_umol_flux(foa, delta).sel(lon=0.0, lat=0.0).item() < 0


def test_to_umol_flux_global_magnitude_is_order_1e_minus_2():
    """Global-mean Foa ~16 mol C m-2 yr-1 with Delta_odis ~-30 permil gives
    ~-1.5e-2 umol C_D14C m-2 s-1."""
    from ocean14c.isoflux import to_umol_flux

    foa = _field([[16.0]], [0.0], [0.0])
    delta = _field([[-30.0]], [0.0], [0.0])
    value = abs(to_umol_flux(foa, delta).sel(lon=0.0, lat=0.0).item())
    assert 1e-3 < value < 1e-1


def test_to_umol_flux_round_trips_to_pgc_permil_per_yr():
    """Converting the shipped flux back through the PgC permil/yr chain must
    reproduce compute_isoflux() exactly -- the guard against a stray factor."""
    from ocean14c.foa import foa_flux_to_pgc_per_yr
    from ocean14c.isoflux import SECONDS_PER_YEAR, compute_isoflux, to_umol_flux

    foa = _field([[16.0, 12.0]], [0.0, 30.0], [0.0])
    delta = _field([[-50.0, -80.0]], [0.0, 30.0], [0.0])
    area = 1.23e10  # m^2

    direct = compute_isoflux(foa_flux_to_pgc_per_yr(foa, area), delta)
    back = to_umol_flux(foa, delta) * 1e-6 * 1000.0 * SECONDS_PER_YEAR * area * 12.011 * 1e-15

    assert np.allclose(back.values, direct.values, rtol=1e-12)
