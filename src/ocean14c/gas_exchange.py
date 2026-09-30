"""Gas-transfer velocity for CO2 across the air-sea interface.

Schmidt-number polynomials and the wind-speed dependent transfer velocity
k, per Wanninkhof (1992) and Wanninkhof (2014). See CLAUDE.md Sec 4.
"""

# Wanninkhof (2014) Table 1, CO2 in seawater (35 permil), -2 to 40 C:
# Sc = A + B*t + C*t^2 + D*t^3 + E*t^4
_SC_COEFFS_2014 = (2116.8, -136.25, 4.7353, -0.092307, 0.0007555)

# Wanninkhof (1992) Table A1, CO2 in seawater (35 permil), 0 to 30 C:
# Sc = A - B*t + C*t^2 - D*t^3
_SC_COEFFS_1992 = (2073.1, 125.62, 3.6276, 0.043219)


def schmidt_number(temperature_c: float, poly: str = "2014"):
    """Schmidt number of CO2 in seawater at the given temperature (deg C).

    poly: "2014" (Wanninkhof 2014 quartic fit, valid -2 to 40 C) or
          "1992" (Wanninkhof 1992 cubic fit, valid 0 to 30 C).
    """
    t = temperature_c
    if poly == "2014":
        a, b, c, d, e = _SC_COEFFS_2014
        return a + b * t + c * t**2 + d * t**3 + e * t**4
    elif poly == "1992":
        a, b, c, d = _SC_COEFFS_1992
        return a - b * t + c * t**2 - d * t**3
    else:
        raise ValueError(f"unknown Schmidt-number polynomial: {poly!r}")


# Default gamma: Wanninkhof (1992) Eq. 3, calibrated for the "steady wind"
# case, i.e. the true second moment <U^2> rather than the square of the mean
# wind speed -- see CLAUDE.md Sec 4.3 and the plan's gas-transfer discussion.
DEFAULT_GAMMA = 0.31


def transfer_velocity(u2, sc, gamma: float = DEFAULT_GAMMA):
    """Gas-transfer velocity k [cm/hr] = gamma * <U^2> * (Sc/660)^-0.5.

    u2: mean of squared wind speed, <U^2>, in (m/s)^2.
    sc: Schmidt number of CO2 at the local sea-surface temperature.
    gamma: calibration constant (default: Wanninkhof 1992 Eq. 3, 0.31).
    """
    return gamma * u2 * (sc / 660.0) ** -0.5
