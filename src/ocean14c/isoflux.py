"""Combine Foa and Delta_odis into the isoflux and aggregate (CLAUDE.md Sec 5).

isoflux = Foa * Delta_odis, per grid cell (and, upstream, per month before
annual aggregation). Table 1 is built from straight sums of the gridded
monthly product; Table S2 reports flux-weighted spatial+temporal means
(Delta_odis_mean = sum(Foa*Delta_odis)/sum(Foa), Foa_mean = sum(Foa)/N) --
both aggregators are implemented here so validate.py can check each table
against the aggregator it actually uses.
"""

import xarray as xr

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0
PERMIL_TO_FRACTION = 1e-3
MOL_TO_UMOL = 1e6


def compute_isoflux(foa: xr.DataArray, delta_odis: xr.DataArray) -> xr.DataArray:
    """isoflux = Foa * Delta_odis, elementwise."""
    return foa * delta_odis


def to_umol_flux(foa_mol_per_m2_yr: xr.DataArray, delta_odis_permil: xr.DataArray) -> xr.DataArray:
    """C_D14C flux [umol C_D14C m^-2 s^-1] -- the field TM5 consumes.

    TM5's radio_co2 setup (Basu et al. 2016) carries the composite tracer
        C_D14C = CO2 x Delta14C,
    with Delta14C a DIMENSIONLESS fraction (permil / 1000). The ocean
    disequilibrium term is a genuine isoflux, Foa x Delta_odis, so the
    conversion is a straight restatement of that product in the group's
    flux units:

        foa [mol C m^-2 yr^-1]
          / SECONDS_PER_YEAR    -> mol C m^-2 s^-1
          x delta_odis / 1000   -> mol C_D14C m^-2 s^-1  (dimensionless Delta)
          x 1e6                 -> umol C_D14C m^-2 s^-1

    NOTE the /1000 is real here: this term is a flux times a per-mil
    difference, not a bare 14C source.
    There, production is a source of bare 14C atoms converted via 1/R_std,
    which lands in the dimensionless-Delta convention already. Here
    Delta_odis is carried explicitly in permil all the way through (that is
    what Table 1 / Table S2 report), so it must be divided by 1000 to become
    the dimensionless Delta that C_D14C is defined with.

    Sign: Delta_odis is negative over essentially the whole ocean for
    2000-2012, so this flux is negative -- the ocean is a 14C sink in
    isoflux terms even though Foa itself is a positive (outgassing) flux.
    """
    return (
        foa_mol_per_m2_yr
        / SECONDS_PER_YEAR
        * (delta_odis_permil * PERMIL_TO_FRACTION)
        * MOL_TO_UMOL
    )


def spatial_sum(field: xr.DataArray, mask: xr.DataArray = None) -> float:
    """Sum over all (lat, lon) cells, optionally restricted to `mask`, NaN-safe."""
    if mask is not None:
        field = field.where(mask)
    return float(field.sum(skipna=True))


def flux_weighted_mean(delta_odis: xr.DataArray, foa: xr.DataArray, mask: xr.DataArray = None) -> float:
    """sum(Foa*Delta_odis) / sum(Foa), Table S2's definition of the reported
    mean Delta_odis (and, symmetrically, usable for a flux-weighted Foa)."""
    if mask is not None:
        delta_odis = delta_odis.where(mask)
        foa = foa.where(mask)
    numerator = float((foa * delta_odis).sum(skipna=True))
    denominator = float(foa.sum(skipna=True))
    return numerator / denominator
