import numpy as np
import pytest

from ocean14c.regression import COEFFICIENTS, rate_of_change


def test_original_coefficients_match_claude_md():
    m_cold, b_cold = COEFFICIENTS["original"]["cold"]
    m_warm, b_warm = COEFFICIENTS["original"]["warm"]
    assert m_cold == pytest.approx(-0.0200)
    assert b_cold == pytest.approx(-3.0851)
    assert m_warm == pytest.approx(-0.0549)
    assert b_warm == pytest.approx(2.9832)


def test_v2_coefficients_match_claude_md():
    m_cold, b_cold = COEFFICIENTS["v2"]["cold"]
    m_warm, b_warm = COEFFICIENTS["v2"]["warm"]
    assert m_cold == pytest.approx(-0.0146)
    assert b_cold == pytest.approx(-3.1359)
    assert m_warm == pytest.approx(-0.0543)
    assert b_warm == pytest.approx(2.8324)


def test_rate_of_change_dispatches_cold_vs_warm_population():
    delta_1995 = np.array([-100.0, 50.0])
    is_cold = np.array([True, False])
    rate = rate_of_change(delta_1995, is_cold, version="original")
    expected_cold = -0.0200 * -100.0 + -3.0851
    expected_warm = -0.0549 * 50.0 + 2.9832
    assert rate[0] == pytest.approx(expected_cold)
    assert rate[1] == pytest.approx(expected_warm)


def test_rate_of_change_unknown_version_raises():
    with pytest.raises(ValueError):
        rate_of_change(np.array([0.0]), np.array([True]), version="bogus")


# --- northern North Pacific correction (docs/SCIENCE.md 8.2) ---------------

def test_glodap2020_npac_leaves_cold_and_warm_identical_to_original():
    """Only the northern North Pacific is revised; the published cold/warm
    coefficients validated against independent GLODAP data and are unchanged."""
    from ocean14c.regression import COEFFICIENTS

    for key in ("cold", "warm"):
        assert COEFFICIENTS["glodap2020_npac"][key] == COEFFICIENTS["original"][key]


def test_npac_population_overrides_cold_only_where_masked():
    import numpy as np

    from ocean14c.regression import COEFFICIENTS, rate_of_change

    delta = np.array([100.0, 100.0])
    is_cold = np.array([True, True])
    is_npac = np.array([True, False])

    out = rate_of_change(delta, is_cold, version="glodap2020_npac", is_npac=is_npac)
    m_np, b_np = COEFFICIENTS["glodap2020_npac"]["npac"]
    m_c, b_c = COEFFICIENTS["glodap2020_npac"]["cold"]
    assert out[0] == pytest.approx(m_np * 100.0 + b_np)
    assert out[1] == pytest.approx(m_c * 100.0 + b_c)


def test_two_population_versions_ignore_the_npac_mask():
    """Passing is_npac must not change 'original'/'v2' results at all --
    the default reproduction of Table 1/S2 has to stay bit-for-bit."""
    import numpy as np

    from ocean14c.regression import rate_of_change

    delta = np.array([100.0, -50.0, 20.0])
    is_cold = np.array([True, True, False])
    is_npac = np.array([True, False, True])

    for version in ("original", "v2"):
        without = rate_of_change(delta, is_cold, version=version)
        with_mask = rate_of_change(delta, is_cold, version=version, is_npac=is_npac)
        assert np.array_equal(without, with_mask)


def test_npac_refit_predicts_observed_north_pacific_rate_better():
    """At a representative northern N. Pacific Delta14C_1995 of ~+60 permil,
    the refit must land near the observed ~-20 permil/decade, where the
    published cold line gives roughly twice that."""
    from ocean14c.regression import COEFFICIENTS

    d95 = 60.0
    m_c, b_c = COEFFICIENTS["original"]["cold"]
    m_np, b_np = COEFFICIENTS["glodap2020_npac"]["npac"]

    published = (m_c * d95 + b_c) * 10.0
    refit = (m_np * d95 + b_np) * 10.0

    assert published < -40.0            # published: steep decline
    assert -25.0 < refit < -15.0        # refit: near the observed -20
