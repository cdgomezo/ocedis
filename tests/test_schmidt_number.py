import pytest

from ocean14c.gas_exchange import schmidt_number, transfer_velocity


def test_schmidt_number_20c_wanninkhof2014_matches_published_value():
    # Wanninkhof (2014) Table 1: Sc(CO2, seawater, 20C) = 668
    assert schmidt_number(20.0, poly="2014") == pytest.approx(668, abs=1)


def test_schmidt_number_20c_wanninkhof1992_close_to_2014_value():
    # Wanninkhof (1992) Table A1 CO2/seawater 3rd-order fit; the two
    # polynomials should agree closely at 20C even though coefficients differ.
    assert schmidt_number(20.0, poly="1992") == pytest.approx(668, abs=15)


def test_schmidt_number_decreases_with_temperature():
    assert schmidt_number(0.0, poly="2014") > schmidt_number(30.0, poly="2014")


def test_schmidt_number_invalid_poly_raises():
    with pytest.raises(ValueError):
        schmidt_number(20.0, poly="bogus")


def test_transfer_velocity_at_sc660_reduces_to_gamma_times_u2():
    # at Sc == 660, (Sc/660)^-0.5 == 1, so k == gamma * U2 exactly
    k = transfer_velocity(u2=50.0, sc=660.0, gamma=0.31)
    assert k == pytest.approx(0.31 * 50.0)


def test_transfer_velocity_scales_with_gamma():
    k_low = transfer_velocity(u2=50.0, sc=660.0, gamma=0.2)
    k_high = transfer_velocity(u2=50.0, sc=660.0, gamma=0.4)
    assert k_high == pytest.approx(2 * k_low)


def test_transfer_velocity_higher_schmidt_number_reduces_k():
    k_sc660 = transfer_velocity(u2=50.0, sc=660.0, gamma=0.31)
    k_sc1000 = transfer_velocity(u2=50.0, sc=1000.0, gamma=0.31)
    assert k_sc1000 < k_sc660
