from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ocean14c.validate import (
    TABLE_1_FOA_DELTA_ODIS,
    TABLE_1_SUM,
    TABLE_S2,
    compare_series,
    fit_gamma,
    load_ocediseq_annual,
    spatial_correlation,
)

OCEDISEQ_PATH = Path(__file__).resolve().parent.parent / "Ocediseq.nc"
requires_ocediseq = pytest.mark.skipif(
    not OCEDISEQ_PATH.exists(), reason="Ocediseq.nc ground-truth file not present"
)


def test_table1_matches_claude_md_endpoints_and_sum():
    assert TABLE_1_FOA_DELTA_ODIS[2000] == pytest.approx(-3426.6)
    assert TABLE_1_FOA_DELTA_ODIS[2012] == pytest.approx(-1303.3)
    assert len(TABLE_1_FOA_DELTA_ODIS) == 13
    assert TABLE_1_SUM == pytest.approx(-29683, abs=1)
    assert sum(TABLE_1_FOA_DELTA_ODIS.values()) == pytest.approx(TABLE_1_SUM, rel=1e-3)


def test_table_s2_matches_claude_md_endpoints():
    assert TABLE_S2[2000]["global_foa"] == pytest.approx(69.1)
    assert TABLE_S2[2000]["global_delta_odis"] == pytest.approx(-51.4)
    assert TABLE_S2[2000]["socean_foa"] == pytest.approx(21.3)
    assert TABLE_S2[2000]["socean_delta_odis"] == pytest.approx(-114.4)
    assert TABLE_S2[2012]["global_foa"] == pytest.approx(74.6)
    assert len(TABLE_S2) == 13


def test_compare_series_reports_percent_error():
    computed = {2000: -3400.0, 2001: -3300.0}
    target = {2000: -3426.6, 2001: -3271.2}
    result = compare_series(computed, target)
    assert result[2000]["computed"] == pytest.approx(-3400.0)
    assert result[2000]["target"] == pytest.approx(-3426.6)
    assert result[2000]["pct_error"] == pytest.approx(
        100 * (-3400.0 - -3426.6) / -3426.6
    )


def test_spatial_correlation_is_1_for_identical_fields():
    a = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("lat", "lon"))
    assert spatial_correlation(a, a) == pytest.approx(1.0)


def test_spatial_correlation_ignores_nan_and_is_scale_invariant():
    a = xr.DataArray([[1.0, 2.0], [3.0, np.nan]], dims=("lat", "lon"))
    b = xr.DataArray([[10.0, 20.0], [30.0, 999.0]], dims=("lat", "lon"))  # 999 where a is NaN
    assert spatial_correlation(a, b) == pytest.approx(1.0)


@requires_ocediseq
def test_load_ocediseq_annual_2006_has_correct_grid_and_plausible_sum():
    field = load_ocediseq_annual(OCEDISEQ_PATH, 2006)
    assert set(field.dims) == {"lat", "lon"}
    assert field["lat"].min().item() == pytest.approx(-89.5)
    assert field["lon"].max().item() == pytest.approx(179.5)
    total = float(field.sum(skipna=True))
    # Same order of magnitude and sign as Table 1's published 2006 value
    # (not identical -- this is the paper's own gridded release, which is
    # close to but not exactly Table 1; see docs/DATA.md).
    assert total < 0
    assert total == pytest.approx(TABLE_1_FOA_DELTA_ODIS[2006], rel=0.1)


def test_fit_gamma_recovers_known_gamma_from_synthetic_linear_target():
    # Foa is linear in gamma (k is linear in gamma), so a target series
    # generated at a known gamma should be recovered exactly.
    true_gamma = 0.27

    def foa_at_gamma(gamma):
        scale = gamma / true_gamma
        return {2000: 69.1 * scale, 2001: 69.2 * scale}

    fitted = fit_gamma(foa_at_gamma, target={2000: 69.1, 2001: 69.2}, bounds=(0.1, 0.6))
    assert fitted == pytest.approx(true_gamma, abs=1e-3)
