import numpy as np
import pytest
import xarray as xr

from ocean14c.atmosphere import (
    delta_atm_field,
    fit_year_edge_values,
    splice_station_series,
)


def _synthetic_series(n_years=10, start_year=2000, trend_per_year=-5.0, seasonal_amp=3.0):
    t = np.linspace(start_year, start_year + n_years, 400)
    values = trend_per_year * (t - start_year) + 100.0 + seasonal_amp * np.sin(2 * np.pi * t)
    rng = np.random.default_rng(0)
    values = values + rng.normal(scale=0.2, size=t.shape)
    return list(zip(t.tolist(), values.tolist()))


def test_fit_year_edge_values_recovers_linear_trend_at_jan1():
    series = _synthetic_series(n_years=8, start_year=2000, trend_per_year=-5.0, seasonal_amp=0.0)
    years = [2002, 2003, 2004, 2005]
    edges = fit_year_edge_values(series, years)
    for year in years:
        expected = -5.0 * (year - 2000) + 100.0
        assert edges[year] == pytest.approx(expected, abs=0.5)


def test_fit_year_edge_values_averages_out_seasonal_cycle_at_jan1():
    # at t = year.0 the pure sin(2*pi*t) seasonal term is ~0, so year-edge
    # values should track the trend line closely despite a seasonal cycle
    series = _synthetic_series(n_years=8, start_year=2000, trend_per_year=-5.0, seasonal_amp=5.0)
    edges = fit_year_edge_values(series, [2003, 2004])
    for year in (2003, 2004):
        expected = -5.0 * (year - 2000) + 100.0
        assert edges[year] == pytest.approx(expected, abs=1.0)


def test_splice_station_series_picks_station_by_year_window():
    station_series = {
        "NWR": _synthetic_series(n_years=13, start_year=2000, trend_per_year=-3.0),
        "JFJ": _synthetic_series(n_years=13, start_year=2000, trend_per_year=-3.5),
    }
    windows = {"NWR": (2004, 2012), "JFJ": (2000, 2003)}
    years = list(range(2000, 2013))

    spliced = splice_station_series(station_series, windows, years)

    assert set(spliced.keys()) == set(years)
    # sanity: values should roughly follow a monotonically declining trend
    assert spliced[2000] > spliced[2012]


def test_splice_station_series_raises_if_year_not_covered_by_any_window():
    station_series = {"NWR": _synthetic_series(n_years=5, start_year=2004)}
    windows = {"NWR": (2004, 2008)}
    with pytest.raises(ValueError):
        splice_station_series(station_series, windows, years=[2000])


def test_delta_atm_field_applies_tropical_band():
    lat = xr.DataArray([20.0, 0.0, -20.0, -50.0], dims="lat", coords={"lat": [20.0, 0.0, -20.0, -50.0]})
    field = delta_atm_field(lat, delta_n=50.0, delta_s=40.0, band_north=10.0, band_south=-40.0, band_offset=10.0)

    assert field.sel(lat=20.0).item() == pytest.approx(50.0)  # north, outside band
    assert field.sel(lat=0.0).item() == pytest.approx(50.0 + 10.0)  # north, in band
    assert field.sel(lat=-20.0).item() == pytest.approx(40.0 + 10.0)  # south, in band
    assert field.sel(lat=-50.0).item() == pytest.approx(40.0)  # south, outside band


# --- multi-period station windows and extrapolation flagging ---------------

def test_window_covers_handles_single_and_multi_range():
    from ocean14c.atmosphere import window_covers

    assert window_covers((2000, 2003), 2001)
    assert not window_covers((2000, 2003), 2004)
    multi = ((2000, 2003), (2021, 2030))
    assert window_covers(multi, 2002)
    assert window_covers(multi, 2025)
    assert not window_covers(multi, 2010)


def test_station_selection_for_the_validation_window_is_unchanged():
    """2000-2012 must resolve exactly as it did before the post-2012 extension,
    or the Table 1 / Table S2 reproduction silently changes."""
    from ocean14c.atmosphere import (
        NORTHERN_HEMISPHERE_WINDOWS,
        SOUTHERN_HEMISPHERE_WINDOWS,
        select_station,
    )

    expected_nh = {y: ("JFJ" if y <= 2003 else "NWR") for y in range(2000, 2013)}
    expected_sh = {y: ("SPO" if y <= 2007 else "DRP") for y in range(2000, 2013)}

    for year in range(2000, 2013):
        assert select_station(NORTHERN_HEMISPHERE_WINDOWS, year) == expected_nh[year]
        assert select_station(SOUTHERN_HEMISPHERE_WINDOWS, year) == expected_sh[year]


def test_every_year_through_2025_has_a_station_in_both_hemispheres():
    from ocean14c.atmosphere import (
        NORTHERN_HEMISPHERE_WINDOWS,
        SOUTHERN_HEMISPHERE_WINDOWS,
        select_station,
    )

    for year in range(2000, 2026):
        assert select_station(NORTHERN_HEMISPHERE_WINDOWS, year) is not None
        assert select_station(SOUTHERN_HEMISPHERE_WINDOWS, year) is not None


def test_select_station_returns_none_outside_all_windows():
    from ocean14c.atmosphere import NORTHERN_HEMISPHERE_WINDOWS, select_station

    assert select_station(NORTHERN_HEMISPHERE_WINDOWS, 1975) is None


def test_extrapolation_gap_zero_when_covered_positive_beyond():
    from ocean14c.atmosphere import extrapolation_gap

    series = [(2020.0, 10.0), (2023.5, 5.0)]
    assert extrapolation_gap(series, 2022) == 0.0        # 2022.96 < 2023.5
    assert extrapolation_gap(series, 2023) == pytest.approx(2023 + 11.5 / 12 - 2023.5)
    assert extrapolation_gap(series, 2025) > 1.0
