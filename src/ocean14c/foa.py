"""One-way ocean-to-atmosphere CO2 flux, Foa (CLAUDE.md Sec 4.4-4.5).

Foa = k * s * pCO2_ocean, depending only on the ocean side (as opposed to
the net flux, which would use the ocean-atmosphere pCO2 difference). Follows
the isotope mass-balance convention of Tans (1980).
"""

import numpy as np

from ocean14c.gas_exchange import DEFAULT_GAMMA, schmidt_number, transfer_velocity
from ocean14c.solubility import co2_solubility

# k [cm/hr] -> [m/yr]: (1 m / 100 cm) * (24 hr/day * 365.25 day/yr)
CM_PER_HR_TO_M_PER_YR = 24 * 365.25 / 100
# s [mol/(L*atm)] -> [mol/(m^3*atm)]: 1 m^3 = 1000 L
L_TO_M3 = 1000.0
# pCO2 [uatm] -> [atm]
UATM_TO_ATM = 1e-6

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0  # for PgC/yr <-> mol/s style conversions
G_PER_MOL_C = 12.011
PG_PER_G = 1e-15


def foa_flux(k_cm_per_hr, solubility_mol_per_L_atm, pco2_uatm):
    """Foa [mol C / (m^2 * yr)] from k [cm/hr], solubility [mol/(L*atm)],
    and ocean pCO2 [uatm].

    Deliberately does NOT np.asarray() its inputs: doing so would strip
    xarray dim/coordinate metadata and fall back to positional numpy
    broadcasting, which silently breaks (or raises) once inputs with
    different dim orders meet -- e.g. a GLODAP-grid field (dims lon, lat)
    combined with a Takahashi/ERA5-grid field (dims month, lat, lon). Plain
    `*` on xarray DataArrays aligns by dimension name instead, which is what
    every caller actually needs; it also still works fine for plain
    floats/ndarrays.
    """
    k_m_per_yr = k_cm_per_hr * CM_PER_HR_TO_M_PER_YR
    s_mol_per_m3_atm = solubility_mol_per_L_atm * L_TO_M3
    pco2_atm = pco2_uatm * UATM_TO_ATM
    return k_m_per_yr * s_mol_per_m3_atm * pco2_atm


def compute_foa(u2, sst_c, pco2_ocean_uatm, gamma: float = DEFAULT_GAMMA, sc_poly: str = "1992"):
    """Foa [mol C / (m^2 * yr)] from wind, SST, and ocean pCO2 fields."""
    sc = schmidt_number(sst_c, poly=sc_poly)
    k = transfer_velocity(u2, sc, gamma=gamma)
    s = co2_solubility(sst_c)
    return foa_flux(k, s, pco2_ocean_uatm)


def foa_flux_to_pgc_per_yr(foa_mol_per_m2_yr, cell_area_m2):
    """Convert a per-area Foa flux to PgC/yr for one grid cell."""
    return foa_mol_per_m2_yr * cell_area_m2 * G_PER_MOL_C * PG_PER_G
