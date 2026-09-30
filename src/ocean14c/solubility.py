"""CO2 solubility in seawater (Weiss, 1974).

Coefficients from Wanninkhof (1992) Table A2, which also confirms (main
text) that this fit gives K0 = 0.0324 mol/(L*atm) at 20C -- used as this
module's test anchor.
"""

import numpy as np

# ln(K0) = A1 + A2*(100/T) + A3*ln(T/100) + S*(B1 + B2*(T/100) + B3*(T/100)^2)
# T in Kelvin, S in per mille (practical salinity units), K0 in mol/(L*atm).
_A1, _A2, _A3 = -60.2409, 93.4517, 23.3585
_B1, _B2, _B3 = 0.023517, -0.023656, 0.0047036


def co2_solubility(temperature_c, salinity: float = 35.0):
    """CO2 solubility K0 [mol/(L*atm)] via Weiss (1974).

    Deliberately does NOT np.asarray() `temperature_c` -- see foa.foa_flux's
    docstring for why: it would strip xarray dim/coordinate metadata and
    break dim-name-based broadcasting downstream. Plain arithmetic and
    np.log/np.exp preserve xarray DataArrays via their __array_ufunc__
    protocol, and still work fine for plain floats/ndarrays.
    """
    t_kelvin = temperature_c + 273.15
    t100 = t_kelvin / 100.0
    ln_k0 = (
        _A1
        + _A2 * (100.0 / t_kelvin)
        + _A3 * np.log(t100)
        + salinity * (_B1 + _B2 * t100 + _B3 * t100**2)
    )
    return np.exp(ln_k0)
