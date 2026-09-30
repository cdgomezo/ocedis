"""Table 1-matched flux variant (ocean14c.table1).

The point of this variant is that it changes the amplitude and NOTHING else,
so most of these tests are about what is preserved, not about the arithmetic.
"""

import numpy as np
import pytest
import xarray as xr

from ocean14c import table1
from ocean14c.validate import TABLE_1_FOA_DELTA_ODIS


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


def test_scale_factor_hits_table_1_exactly():
    recon = -1239.1785804681215  # 2011, full 2000-2012 run of 2026-08-18
    f = table1.scale_factor(2011, recon)
    assert recon * f == pytest.approx(TABLE_1_FOA_DELTA_ODIS[2011], rel=1e-12)


def test_scale_factor_is_none_outside_the_table():
    assert table1.scale_factor(2015, -749.79) is None


def test_out_of_table_years_take_the_mean_of_the_in_table_ones():
    isoflux = {2011: -1239.1785804681215, 2012: -1124.08053022371, 2015: -749.79}
    f = table1.scale_factors(isoflux)
    assert f[2015] == pytest.approx((f[2011] + f[2012]) / 2)


def test_falls_back_to_the_frozen_factor_when_no_year_is_covered():
    isoflux = {2015: -749.79, 2020: -161.69}
    f = table1.scale_factors(isoflux)
    assert f[2015] == pytest.approx(table1.REFERENCE_MEAN_SCALE["simplified"])
    assert table1.uses_reference_fallback(isoflux)


def test_the_fallback_follows_the_curve_fit():
    """simplified and ccgcrv differ by up to 4.6% in a single year -- more than
    the spread within either series, so one frozen number for both would be
    wrong for whichever run did not produce it."""
    isoflux = {2015: -749.79}
    assert table1.scale_factors(isoflux, curve_fit="ccgcrv")[2015] == pytest.approx(
        table1.REFERENCE_MEAN_SCALE["ccgcrv"]
    )
    assert table1.REFERENCE_MEAN_SCALE["ccgcrv"] != table1.REFERENCE_MEAN_SCALE["simplified"]


def test_unknown_curve_fit_uses_the_default_rather_than_raising():
    assert table1.reference_mean_scale("something_new") == pytest.approx(
        table1.REFERENCE_MEAN_SCALE[table1.DEFAULT_CURVE_FIT]
    )


def test_in_table_year_disables_the_fallback_warning():
    assert not table1.uses_reference_fallback({2011: -1239.2, 2015: -749.8})


def test_reference_factors_match_the_reference_series():
    """The frozen fallback must stay consistent with Table 1 -- it is only ever
    used when nothing in the run can be recomputed, so nothing else would catch
    it drifting."""
    recon = {  # global_isoflux from full 2000-2012 runs, gamma=0.2146973601735
        "simplified": {2000: -3027.004487, 2006: -1994.915701, 2012: -1124.080530},
        "ccgcrv": {2000: -3053.186709, 2006: -2040.250283, 2012: -1170.264081},
    }
    for curve_fit, series in recon.items():
        for year, value in series.items():
            assert table1.REFERENCE_SCALE_FACTORS[curve_fit][year] == pytest.approx(
                table1.scale_factor(year, value), abs=5e-5
            )


def test_sign_flip_is_refused_not_silently_shipped():
    """A positive reconstruction against a negative target would give a
    negative factor -- an inside-out field, invisible in every aggregate."""
    with pytest.raises(ValueError, match="opposite sign"):
        table1.scale_factor(2011, +1239.2)


def test_zero_and_nan_are_refused():
    with pytest.raises(ValueError):
        table1.scale_factor(2011, 0.0)
    with pytest.raises(ValueError):
        table1.scale_factor(2011, np.nan)


def test_scale_flux_preserves_pattern_dims_coords_and_nan_mask():
    flux = _flux()
    flux.values[:] = np.random.default_rng(0).normal(-2e-2, 5e-3, flux.shape)
    flux.values[:, 0, 0] = np.nan

    scaled = table1.scale_flux(flux, 1.1093)

    assert scaled.dims == flux.dims
    assert list(scaled["lat"].values) == list(flux["lat"].values)
    assert np.isnan(scaled.values[:, 0, 0]).all()
    # amplitude only: every finite cell scaled by the same number, so the
    # correlation with the original is exactly 1 and the sign never moves
    ok = np.isfinite(flux.values)
    assert np.corrcoef(flux.values[ok], scaled.values[ok])[0, 1] == pytest.approx(1.0)
    assert np.all(np.sign(scaled.values[ok]) == np.sign(flux.values[ok]))


def test_the_two_variants_have_distinct_terms():
    """They are written into the same directory; equal terms would mean the
    Table 1 variant silently overwriting the reconstruction."""
    assert table1.TERM_RECONSTRUCTION != table1.TERM_TABLE1
