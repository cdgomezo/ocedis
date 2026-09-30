"""Hua et al. (2021) zonal curves.

The assertions here are the three claims the plan rests on: the files are
F14C (not permil), the five zones collapse to two after 1973, and the
post-2019 extrapolation is exponential in F14C.
"""

import numpy as np
import pytest

from ocean14c import hua


@pytest.fixture(scope="module")
def series():
    return hua.hemispheric_series()


def test_files_are_f14c_and_are_converted_to_permil(series):
    """Paper Section "Results": the last data point, 2019.375, has F14C
    1.0084 (NH) and 1.0195 (SH). Read as permil these would be ~0.001."""
    assert hua.value_at(2019.375, "NH", series=series)[0] == pytest.approx(8.4, abs=0.05)
    assert hua.value_at(2019.375, "SH", series=series)[0] == pytest.approx(19.5, abs=0.05)


def test_curves_span_the_published_record(series):
    t_nh, _ = series["NH"]
    assert t_nh[0] == pytest.approx(1941.2083, abs=1e-3)
    assert t_nh[-1] == pytest.approx(hua.LAST_DATA_YEAR, abs=1e-3)
    # Monthly: 1/12 yr steps.
    assert np.allclose(np.diff(t_nh), 1.0 / 12.0, atol=2e-4)


def test_zones_collapse_to_two_after_1973():
    """The load-bearing claim: post-1973 Hua pools records by hemisphere, so
    resolving five zones in the modern era resolves rounding noise."""
    curves = hua.load_zonal_curves()
    t = curves["NHZ1"][0]
    sel = t >= 2011.0

    for a, b in (("NHZ1", "NHZ2"), ("NHZ1", "NHZ3"), ("NHZ2", "NHZ3")):
        assert np.max(np.abs(curves[a][1][sel] - curves[b][1][sel])) < 0.5
    ts = curves["SHZ3"][0]
    ssel = ts >= 2011.0
    assert np.max(np.abs(curves["SHZ3"][1][ssel] - curves["SHZ1_2"][1][ssel])) < 0.5

    # ... while the hemispheres genuinely differ.
    assert np.max(np.abs(curves["NHZ1"][1][sel][:36] - curves["SHZ3"][1][ssel][:36])) > 4.0


def test_zones_are_far_apart_at_the_bomb_peak():
    """Sanity that the collapse above is a real 1973 change, not a parser bug
    flattening everything."""
    curves = hua.load_zonal_curves()
    t = curves["NHZ1"][0]
    sel = (t >= 1963.0) & (t < 1966.0)
    assert np.max(np.abs(curves["NHZ1"][1][sel] - curves["NHZ3"][1][sel])) > 100.0


def test_zone_of_refuses_the_five_zone_era():
    with pytest.raises(NotImplementedError, match="Figure 1"):
        hua.zone_of(20.0, lon=0.0, t=1965.0)


def test_zone_of_splits_at_the_equator():
    assert hua.zone_of(46.5, t=2012.0) == "NH"      # Jungfraujoch
    assert hua.zone_of(-41.4, t=2012.0) == "SH"     # Baring Head
    assert hua.zone_of(0.0, t=2012.0) == "NH"       # boundary belongs north


def test_raw_extrapolation_does_not_land_on_the_papers_own_record_end():
    """The Appendix exponentials are least-squares fits over 1993/1994-2019
    that were never constrained to be continuous at the join, so evaluated at
    2019.375 they miss the tabulated F14C by 1.8 (NH) and 3.6 (SH) permil.

    Pinned because it is a property of the published coefficients, not of this
    code: if these numbers move, the coefficients were mistyped."""
    assert hua.extrapolate_f14c(hua.LAST_DATA_YEAR, "NH", anchor=False) == pytest.approx(
        1.006642, abs=1e-5
    )
    assert hua.extrapolate_f14c(hua.LAST_DATA_YEAR, "SH", anchor=False) == pytest.approx(
        1.015888, abs=1e-5
    )


def test_anchored_extrapolation_is_continuous_with_the_record():
    """The default. Rescaling is multiplicative, so the decay rate -- the only
    thing the fit is really for -- survives."""
    for hemisphere, expected in (("NH", 8.4), ("SH", 19.5)):
        f = hua.extrapolate_f14c(hua.LAST_DATA_YEAR, hemisphere)
        assert (f - 1.0) * 1000.0 == pytest.approx(expected, abs=0.01)

    # Decay rate preserved: the ratio over one year is the raw fit's ratio.
    raw = hua.extrapolate_f14c(np.array([2020.0, 2021.0]), "NH", anchor=False)
    anc = hua.extrapolate_f14c(np.array([2020.0, 2021.0]), "NH")
    assert anc[1] / anc[0] == pytest.approx(raw[1] / raw[0], rel=1e-12)


def test_extrapolation_refuses_beyond_the_papers_own_limit():
    with pytest.raises(ValueError, match="atmfield"):
        hua.extrapolate_f14c(2030.0, "NH")


def test_value_at_crosses_into_extrapolation_smoothly(series):
    t = np.array([2019.29, 2019.375, 2019.46])
    v = hua.value_at(t, "NH", series=series)
    assert np.all(np.isfinite(v))
    assert np.max(np.abs(np.diff(v))) < 2.0


def test_field_is_a_step_at_the_equator(series):
    lat = np.array([-0.5, 0.5])
    f = hua.field_at(lat, 2012.5, series=series)
    assert f[1] - f[0] == pytest.approx(
        hua.value_at(2012.5, "NH", series=series)[0]
        - hua.value_at(2012.5, "SH", series=series)[0]
    )


def test_blending_removes_the_discontinuity(series):
    lat = np.linspace(-30, 30, 601)
    hard = hua.field_at(lat, 2012.5, series=series)
    soft = hua.field_at(lat, 2012.5, series=series, blend_deg=10.0)
    assert np.max(np.abs(np.diff(hard))) > 2.0
    assert np.max(np.abs(np.diff(soft))) < 0.2
    # Blending is a redistribution, not a shift.
    assert hua.area_weighted_mean(lat, soft) == pytest.approx(
        hua.area_weighted_mean(lat, hard), abs=0.3
    )


def test_anomaly_level_keeps_shape_and_moves_level(series):
    lat = np.linspace(-89.5, 89.5, 180)
    absolute = hua.field_at(lat, 2012.5, series=series)
    anomaly = hua.field_at(lat, 2012.5, series=series, level="anomaly", global_mean=0.0)

    assert hua.area_weighted_mean(lat, anomaly) == pytest.approx(0.0, abs=1e-9)
    # Same contrast, different offset.
    assert np.ptp(anomaly) == pytest.approx(np.ptp(absolute), abs=1e-9)


def test_anomaly_level_requires_a_level():
    with pytest.raises(ValueError, match="global_mean"):
        hua.field_at(np.array([0.0]), 2012.5, level="anomaly")


def test_hua_sits_above_graven_over_2011_2013(series):
    """Documents the ~6-8 permil offset that motivates level="anomaly".
    Graven CMIP6 annual NH: 2011 37.3, 2012 30.4, 2013 24.3."""
    graven_nh = {2011: 37.3, 2012: 30.4, 2013: 24.3}
    for year, g in graven_nh.items():
        months = year + (np.arange(12) + 0.5) / 12.0
        ours = float(np.mean(hua.value_at(months, "NH", series=series)))
        assert 3.0 < ours - g < 12.0
