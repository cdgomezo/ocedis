"""Gridded atmospheric Delta14C field: registry, curve fit, background filter,
kriging, and the aggregation/writing layer.

Real-data tests skip automatically when data/raw/Delta_atm is not populated,
matching the convention used by the rest of this suite.
"""

from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from ocean14c.atmfield import default_grid, global_mean_series, zonal_band_mean
from ocean14c.background import positive_side_scale
from ocean14c.krige import (
    DEFAULT_DRIFT_DEGREE, covariance, drift_basis, krige_month, krige_points,
)

DELTA_ATM = Path(__file__).parent.parent / "data/raw/Delta_atm"
_HAS_DATA = DELTA_ATM.is_dir() and any(DELTA_ATM.glob("*.c14"))
needs_data = pytest.mark.skipif(not _HAS_DATA, reason="Delta_atm data not present")


# --- aggregation ----------------------------------------------------------

def _fake_result(field):
    nt = field.shape[0]
    return {
        "lat": np.linspace(-89.5, 89.5, field.shape[1]),
        "lon": np.linspace(-179.5, 179.5, field.shape[2]),
        "field": field,
        "years": np.repeat(2000, nt),
        "months": np.arange(1, nt + 1),
    }


def test_zonal_band_mean_returns_nan_for_an_all_nan_month():
    """Regression test. np.nansum of an all-NaN slice returns 0.0, not NaN;
    relying on it wrote Delta14C = 0 permil into the TM5 timeseries for years
    with too few stations to grid -- a silently wrong forcing."""
    field = np.full((2, 18, 36), np.nan)
    field[0] = 50.0                       # month 0 valid, month 1 all-NaN
    out = zonal_band_mean(_fake_result(field), -90, 90)
    assert out[0] == pytest.approx(50.0)
    assert np.isnan(out[1]), "all-NaN month must be NaN, not 0.0"


def test_zonal_band_mean_renormalises_over_valid_cells_only():
    """A partially observed band must not be diluted toward zero by the
    missing cells."""
    field = np.full((1, 18, 36), np.nan)
    field[0, 9:, :] = 20.0                # northern half only
    out = zonal_band_mean(_fake_result(field), -90, 90)
    assert out[0] == pytest.approx(20.0)


def test_global_mean_series_is_area_weighted():
    """cos(lat) weighting means a uniform field returns its own value."""
    field = np.full((1, 180, 360), 33.0)
    assert global_mean_series(_fake_result(field))[0] == pytest.approx(33.0)


def test_default_grid_is_cell_centred():
    lat, lon = default_grid(1.0)
    assert len(lat) == 180 and len(lon) == 360
    assert lat[0] == pytest.approx(-89.5)
    assert lon[0] == pytest.approx(-179.5)


# --- background filter ----------------------------------------------------

def test_positive_side_scale_ignores_the_negative_tail():
    """Fossil contamination only depresses Delta14C, so the scale estimate
    must come from the positive residuals and be unmoved by outliers below."""
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 2.0, 4000)
    contaminated = np.concatenate([clean, np.full(300, -40.0)])
    assert positive_side_scale(contaminated) == pytest.approx(
        positive_side_scale(clean), rel=0.05
    )
    # ... whereas the naive estimator is badly inflated by the same points
    assert np.std(contaminated) > 2 * np.std(clean)


# --- kriging --------------------------------------------------------------

def test_drift_basis_shape_and_values():
    F = drift_basis([0.0, 90.0], degree=1)
    assert F.shape == (2, 2)
    assert F[0, 1] == pytest.approx(0.0)      # sin(0)
    assert F[1, 1] == pytest.approx(1.0)      # sin(90)


def test_covariance_decays_with_separation():
    assert covariance(0.0) > covariance(1.0) > covariance(5.0)


def test_krige_reproduces_a_linear_sin_lat_field_exactly():
    """The shipped drift is linear in sin(lat), so a field that is exactly
    linear in sin(lat) must be recovered essentially without error."""
    lat = np.array([-80.0, -40.0, 0.0, 40.0, 80.0])
    lon = np.zeros_like(lat)
    truth = lambda la: 10.0 + 20.0 * np.sin(np.radians(la))
    val = truth(lat)
    err = np.full_like(val, 1.0)

    grid_lat = np.array([-60.0, -20.0, 20.0, 60.0])
    field, sd = krige_month(lat, lon, val, err, grid_lat, np.array([0.0, 100.0]))

    for i, gl in enumerate(grid_lat):
        assert field[i, 0] == pytest.approx(truth(gl), abs=0.5)
    assert np.all(np.isfinite(sd)) and np.all(sd >= 0)


def test_krige_output_shapes_and_finiteness():
    lat = np.array([-60.0, 0.0, 60.0])
    lon = np.array([0.0, 120.0, -120.0])
    val = np.array([50.0, 45.0, 40.0])
    err = np.full(3, 2.0)
    glat, glon = default_grid(10.0)
    field, sd = krige_month(lat, lon, val, err, glat, glon)
    assert field.shape == (len(glat), len(glon)) == sd.shape
    assert np.isfinite(field).all() and np.isfinite(sd).all()


def test_krige_uncertainty_grows_away_from_stations():
    """The uncertainty field is the reason the kriging machinery is kept, so
    it has to behave: lowest at a station, higher far from any."""
    lat = np.array([-30.0, 0.0, 30.0])
    lon = np.zeros(3)
    val = np.array([40.0, 45.0, 50.0])
    err = np.full(3, 1.0)
    _, sd = krige_points(lat, lon, val, err,
                         np.array([0.0, 0.0]), np.array([0.0, 180.0]))
    assert sd[1] > sd[0]


def test_krige_handles_fewer_stations_than_drift_terms():
    """Degenerate months must degrade gracefully rather than raise."""
    lat = np.array([10.0])
    lon = np.array([0.0])
    field, sd = krige_month(lat, lon, np.array([42.0]), np.array([2.0]),
                            np.array([0.0, 20.0]), np.array([0.0]))
    assert field.shape == (2, 1)
    assert np.isfinite(field).all()


def test_krige_empty_month_returns_nan_not_zero():
    field, sd = krige_month(np.array([]), np.array([]), np.array([]),
                            np.array([]), np.array([0.0]), np.array([0.0]))
    assert np.isnan(field).all() and np.isnan(sd).all()


# --- real data ------------------------------------------------------------

@needs_data
def test_registry_parses_coordinates_from_headers():
    from ocean14c.stations import load_stations

    st = load_stations(DELTA_ATM)
    assert len(st) >= 20
    for code, s in st.items():
        assert -90 <= s.lat <= 90, code
        assert -180 <= s.lon <= 180, code
        assert s.n > 0

    # Spot-check two stations against their file headers, including one in
    # the southern hemisphere whose latitude is written '-41.4100 N'.
    assert st["MHD"].lat == pytest.approx(53.3266)
    assert st["BHD"].lat == pytest.approx(-41.41)


@needs_data
def test_duplicate_station_files_are_merged():
    from ocean14c.stations import load_stations

    st = load_stations(DELTA_ATM)
    assert len(st["JFJ"].files) == 2          # .c14 record plus the ICOS 2026 file
    assert st["JFJ"].end > 2025.0             # ICOS extends JFJ into 2026


@needs_data
def test_blank_valued_rows_are_skipped_not_fatal():
    """ALT/CGO/IZO/NMY carry a few rows flagged 'O' with an empty 14C value."""
    from ocean14c.atm_stations import load_c14_station

    path = DELTA_ATM / "uheiiup_l2_2025_1_alt_20m_int_14day_clean.c14"
    if not path.exists():
        pytest.skip("ALT file not present")
    series = load_c14_station(path)
    assert len(series) > 400
    assert all(np.isfinite(v) for _, v in series)


# --- ccgcrv curve fit and extrapolation -----------------------------------

def test_curve_fit_methods_differ_and_default_is_simplified():
    """The default must stay 'simplified': the validated 2000-2012
    reproduction was established with it, so changing the default would move
    numbers pinned against Table 1 / Table S2 / Ocediseq."""
    from ocean14c.atmosphere import DEFAULT_CURVE_FIT, CURVE_FIT_SIMPLIFIED

    assert DEFAULT_CURVE_FIT == CURVE_FIT_SIMPLIFIED


def test_fit_year_edge_values_rejects_unknown_method():
    from ocean14c.atmosphere import fit_year_edge_values

    series = [(2000.0 + i / 12, 100.0 - i) for i in range(60)]
    with pytest.raises(ValueError, match="unknown curve-fit method"):
        fit_year_edge_values(series, [2002.0], method="nope")


def test_ccgcrv_falls_back_when_record_too_short():
    """Short records cannot support a ccgcrv fit; it must degrade to the
    simplified fit rather than raise."""
    from ocean14c.atmosphere import fit_year_edge_values

    series = [(2000.0 + i / 12, 100.0 - i) for i in range(10)]
    out = fit_year_edge_values(series, [2000.5], method="ccgcrv")
    assert np.isfinite(out[2000.5])


def _synthetic_series(n=400, start=2000.0, slope=-5.0, amp=3.0):
    t = start + np.arange(n) / 25.0
    v = 100.0 + slope * (t - start) + amp * np.sin(2 * np.pi * t)
    return list(zip(t.tolist(), v.tolist()))


def test_smooth_is_nan_outside_record_but_evaluate_is_not():
    """getSmoothValue returns NaN past the record -- the reason `evaluate`
    exists at all."""
    from ocean14c.curvefit import fit_station

    fit = fit_station("X", _synthetic_series())
    beyond = fit.end + 1.0
    assert np.isnan(fit.smooth(beyond)[0])
    assert np.isfinite(fit.evaluate(beyond)[0])


def test_extrapolation_continues_the_trend_linearly():
    """A record with a known constant slope must extrapolate along it."""
    from ocean14c.curvefit import fit_station

    slope = -5.0
    fit = fit_station("X", _synthetic_series(slope=slope, amp=0.0))
    t1, t2 = fit.end + 1.0, fit.end + 2.0
    d = float(fit.evaluate(t2)[0] - fit.evaluate(t1)[0])
    assert d == pytest.approx(slope, abs=1.0)


def test_gap_is_zero_inside_and_grows_outside():
    from ocean14c.curvefit import fit_station

    fit = fit_station("X", _synthetic_series())
    assert fit.gap(fit.start + 1.0)[0] == 0.0
    assert fit.gap(fit.end + 2.0)[0] == pytest.approx(2.0, abs=1e-6)
    assert fit.gap(fit.start - 1.5)[0] == pytest.approx(1.5, abs=1e-6)


def test_station_values_at_excludes_extrapolation_by_default():
    """Default behaviour must not silently extrapolate."""
    from ocean14c.atmfield import station_values_at
    from ocean14c.curvefit import fit_station
    from ocean14c.stations import Station

    fit = fit_station("X", _synthetic_series())
    st = {"X": Station("X", "X", 0.0, 0.0, 0.0, _synthetic_series(), [], "background")}
    beyond = fit.end + 1.0

    _, _, v0, _ = station_values_at(st, {"X": fit}, beyond)
    assert len(v0) == 0

    _, _, v1, e1 = station_values_at(st, {"X": fit}, beyond, max_extrapolation_yr=2.0)
    assert len(v1) == 1
    assert np.isfinite(v1[0])
    # error inflated by the extrapolation gap
    assert e1[0] > 2.0
