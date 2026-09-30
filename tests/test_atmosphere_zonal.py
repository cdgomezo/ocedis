"""Zonal geometries added for the Delta_atm comparison.

The load-bearing test is the first one: delta_atm_zonal with geometry
"miller_band" must be delta_atm_field, exactly, or the pinned Table 1 /
Table S2 / Ocediseq reproduction has silently moved.
"""

import numpy as np
import pytest
import xarray as xr

from ocean14c.atmosphere import (
    area_weighted_mean,
    band_of,
    delta_atm_field,
    delta_atm_zonal,
    fit_bands,
    relevel,
)


@pytest.fixture
def lat():
    return xr.DataArray(np.linspace(-89.5, 89.5, 180), dims="lat")


def test_miller_band_geometry_is_the_shipped_field(lat):
    """The regression guard, at the geometry level."""
    old = delta_atm_field(lat, delta_n=30.0, delta_s=40.0).values
    new = delta_atm_zonal(lat, {"NH": 30.0, "SH": 40.0}, "miller_band").values
    assert np.array_equal(old, new)


def test_basu_band_differs_only_where_the_bands_differ(lat):
    bv = {"NH": 30.0, "SH": 40.0}
    miller = delta_atm_zonal(lat, bv, "miller_band").values
    basu = delta_atm_zonal(lat, bv, "basu_band").values
    differ = lat.values[miller != basu]
    # Miller covers 10N..40S, Basu 20N..20S; they agree on the overlap.
    assert differ.min() >= -40.0 and differ.max() <= 20.0
    assert np.any(differ > 10.0) and np.any(differ < -20.0)


def test_hemispheric_is_a_bare_step(lat):
    f = delta_atm_zonal(lat, {"NH": 30.0, "SH": 40.0}, "hemispheric").values
    assert set(np.unique(f)) == {30.0, 40.0}


def test_band_of_assigns_every_latitude():
    lat = np.linspace(-90, 90, 361)
    for geometry in ("hemispheric", "graven"):
        assert not np.any(band_of(lat, geometry) == None)  # noqa: E711


def test_fit_bands_falls_back_when_a_band_is_empty():
    """No station in the tropics must mean "no information", not a fabricated
    contrast."""
    bands = fit_bands([-60.0, 60.0], [40.0, 30.0], "graven")
    assert bands["SH"] == 40.0
    assert bands["NH"] == 30.0
    assert bands["TR"] == pytest.approx(35.0)


def test_blending_is_continuous_and_conserves_the_mean(lat):
    bv = {"NH": 30.0, "SH": 40.0}
    hard = delta_atm_zonal(lat, bv, "hemispheric").values
    soft = delta_atm_zonal(lat, bv, "hemispheric", blend_deg=10.0).values
    assert np.max(np.abs(np.diff(hard))) == pytest.approx(10.0)
    assert np.max(np.abs(np.diff(soft))) < 1.2
    assert area_weighted_mean(lat.values, soft) == pytest.approx(
        area_weighted_mean(lat.values, hard), abs=0.05
    )


def test_relevel_moves_level_without_touching_shape(lat):
    f = delta_atm_zonal(lat, {"NH": 30.0, "SH": 40.0}, "miller_band").values
    g = relevel(lat.values, f, 0.0)
    assert area_weighted_mean(lat.values, g) == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(np.diff(g), np.diff(f))


def test_unknown_geometry_raises(lat):
    with pytest.raises(KeyError):
        delta_atm_zonal(lat, {"NH": 30.0}, "no_such_geometry")


def test_variant_path_reproduces_the_default_path(lat):
    """pipeline.build_delta_atm(variant=None) is the shipped forcing. The
    miller_band variant must be indistinguishable from it, or every number
    this repo has published moved when the variants were added."""
    from ocean14c.pipeline import _monthly_delta_atm, build_delta_atm

    nh = np.linspace(40.0, 38.0, 12)
    sh = np.linspace(46.0, 44.0, 12)
    lon = xr.DataArray(np.linspace(-179.5, 179.5, 360), dims="lon")

    default = _monthly_delta_atm(lat, nh, sh)
    variant = build_delta_atm(lat, lon, 2012, nh, sh,
                              variant={"geometry": "miller_band"})
    assert np.array_equal(default.values, variant.values)
    assert build_delta_atm(lat, lon, 2012, nh, sh, variant=None).equals(default)


def test_level_is_rejected_where_it_has_no_meaning(lat):
    from ocean14c.pipeline import build_delta_atm

    nh, sh = np.full(12, 40.0), np.full(12, 46.0)
    lon = xr.DataArray(np.linspace(-179.5, 179.5, 360), dims="lon")
    with pytest.raises(ValueError, match="our own stations"):
        build_delta_atm(lat, lon, 2012, nh, sh,
                        variant={"geometry": "hemispheric", "level": "stations"})
    with pytest.raises(ValueError, match="no level to choose"):
        build_delta_atm(lat, lon, 2012, nh, sh,
                        variant={"geometry": "krige", "level": "stations"})


def test_hua_shape_variant_takes_our_level(lat):
    """The whole point of level="stations": Hua's contrast, our absolute
    value. Hua sits ~6-8 permil above our network, so if the re-levelling
    silently failed this would be obvious."""
    from ocean14c.pipeline import build_delta_atm

    nh, sh = np.full(12, 40.0), np.full(12, 46.0)
    lon = xr.DataArray(np.linspace(-179.5, 179.5, 360), dims="lon")
    ours = build_delta_atm(lat, lon, 2012, nh, sh, variant={"geometry": "hemispheric"})
    shaped = build_delta_atm(lat, lon, 2012, nh, sh,
                             variant={"geometry": "hua", "level": "stations"})
    absolute = build_delta_atm(lat, lon, 2012, nh, sh,
                               variant={"geometry": "hua", "level": "native"})

    # level="stations" takes its level from the hemispheric values passed in...
    for m in range(12):
        assert area_weighted_mean(lat.values, shaped.values[m]) == pytest.approx(
            area_weighted_mean(lat.values, ours.values[m]), abs=1e-6
        )
    # ...while level="native" ignores them entirely, and level="stations"
    # tracks them one for one. (How far Hua's own level actually sits from
    # ours is a property of the real data, pinned in test_hua.py, not
    # something a synthetic nh/sh can show.)
    shifted = build_delta_atm(lat, lon, 2012, nh + 7.0, sh + 7.0,
                              variant={"geometry": "hua", "level": "stations"})
    shifted_abs = build_delta_atm(lat, lon, 2012, nh + 7.0, sh + 7.0,
                                  variant={"geometry": "hua", "level": "native"})
    assert np.allclose(shifted.values, shaped.values + 7.0)
    assert np.array_equal(shifted_abs.values, absolute.values)

    # And the shape survives the re-levelling: same NH-SH contrast as Hua's own.
    assert np.ptp(shaped.values[0]) == pytest.approx(np.ptp(absolute.values[0]))
