"""Two-population regression of surface ocean d14C rate of change vs. the
1995 GLODAP field (Lindsay 2016 / Miller et al. 2025 SI Fig. S2/S5).

rate [permil/yr] = m * delta_1995 [permil] + b

"cold" = Southern Ocean + northern North Pacific; "warm" = everywhere else
(see ocean14c.regions for the spatial masks).
"""

import numpy as np

# (m [1/yr], b [permil/yr]) per population. CLAUDE.md Sec 3.2 / SI Fig. S2, S5.
#
# A version may optionally carry a third "npac" population for the northern
# North Pacific. When present -- and only then -- it overrides "cold" there.
# Versions without it keep the published two-population behaviour exactly.
COEFFICIENTS = {
    "original": {  # Fig. S2 -- matches Table 1 / Table S2, the primary target
        "cold": (-0.0200, -3.0851),
        "warm": (-0.0549, 2.9832),
    },
    "v2": {  # Fig. S5, GLODAP v2 update -- secondary sensitivity check
        "cold": (-0.0146, -3.1359),
        "warm": (-0.0543, 2.8324),
    },
    # Northern North Pacific correction, fitted from GLODAPv2.2020 bottle
    # Delta14C (1990-2018, <=200 m, 4-degree boxes) by
    # scripts/refit_regression_glodap.py. See docs/SCIENCE.md Sec 8.1/8.2.
    #
    # "cold" and "warm" are UNCHANGED from "original" -- the independent
    # GLODAP test validates the published warm coefficients (bias
    # +2.6 permil/decade over 100 boxes), so only the northern North Pacific
    # is revised. There the published cold line over-predicts the decline by
    # roughly a factor of two (bias -23.2, RMSE 24.6 permil/decade over 24
    # boxes); this refit reduces that to bias 0.0, RMSE 8.0, and is
    # independently corroborated by the NEMO run of Pookkandy et al. (2023).
    #
    # CAVEAT: the fitted slope is likely attenuated by errors-in-variables
    # (Delta14C_1995 carries its own uncertainty) -- the same refit applied
    # to the warm population returns -0.0246 against a published -0.0549.
    # The robust, corroborated part of this correction is the ~ -3 permil/yr
    # intercept offset, not the precise slope. Valid over the fitted range
    # Delta14C_1995 in about [0, 117] permil, which spans the region.
    "glodap2020_npac": {
        "cold": (-0.0200, -3.0851),
        "warm": (-0.0549, 2.9832),
        "npac": (-0.0148, -1.1320),
    },
}


def rate_of_change(delta_1995, is_cold, version: str = "original", is_npac=None):
    """dDelta14C/dt [permil/yr] for each grid cell.

    delta_1995: 1995 GLODAP surface ocean Delta14C field.
    is_cold: boolean mask, True where the cell belongs to the cold/deep
        population (Southern Ocean + northern North Pacific).
    version: "original" (default, matches Table 1/S2), "v2", or
        "glodap2020_npac".
    is_npac: optional boolean mask for the northern North Pacific. Only used
        when `version` defines an "npac" population; ignored otherwise, so
        callers can pass it unconditionally.
    """
    if version not in COEFFICIENTS:
        raise ValueError(f"unknown regression coefficient version: {version!r}")

    delta_1995 = np.asarray(delta_1995, dtype=float)
    is_cold = np.asarray(is_cold, dtype=bool)

    coeffs = COEFFICIENTS[version]
    m_cold, b_cold = coeffs["cold"]
    m_warm, b_warm = coeffs["warm"]

    rate_cold = m_cold * delta_1995 + b_cold
    rate_warm = m_warm * delta_1995 + b_warm
    rate = np.where(is_cold, rate_cold, rate_warm)

    if "npac" in coeffs and is_npac is not None:
        m_np, b_np = coeffs["npac"]
        rate = np.where(np.asarray(is_npac, dtype=bool), m_np * delta_1995 + b_np, rate)

    return rate
