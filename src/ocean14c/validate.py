"""Validation targets and comparison utilities (CLAUDE.md Sec 6).

TABLE_1_FOA_DELTA_ODIS / TABLE_1_SUM: main-text Table 1, global FoaDelta_odis.
TABLE_S2: Supporting Information Table S2, global and Southern Ocean Foa and
flux-weighted mean Delta_odis.
load_ocediseq_annual / spatial_correlation: compare a reconstructed annual
isoflux field against the paper's own published gridded product
(Ocediseq.nc, doi:10.15138/8dhy-6k90) cell by cell, not just as aggregates.
"""

import numpy as np
import xarray as xr
from scipy.optimize import minimize_scalar

# Table 1 (main text): global FoaDelta_odis, PgC permil/yr.
TABLE_1_FOA_DELTA_ODIS = {
    2000: -3426.6,
    2001: -3271.2,
    2002: -2916.0,
    2003: -2769.6,
    2004: -2631.3,
    2005: -2380.7,
    2006: -2278.8,
    2007: -1994.2,
    2008: -1916.4,
    2009: -1661.5,
    2010: -1696.6,
    2011: -1437.2,
    2012: -1303.3,
}
TABLE_1_SUM = -29683

# Table S2 (SI): annual Foa (PgC/yr) and flux-weighted mean Delta_odis (permil),
# global and Southern Ocean.
TABLE_S2 = {
    2000: {"global_foa": 69.1, "global_delta_odis": -51.4, "socean_foa": 21.3, "socean_delta_odis": -114.4},
    2001: {"global_foa": 69.2, "global_delta_odis": -49.0, "socean_foa": 22.1, "socean_delta_odis": -111.8},
    2002: {"global_foa": 69.3, "global_delta_odis": -43.5, "socean_foa": 20.7, "socean_delta_odis": -108.8},
    2003: {"global_foa": 69.4, "global_delta_odis": -41.3, "socean_foa": 21.6, "socean_delta_odis": -104.8},
    2004: {"global_foa": 70.2, "global_delta_odis": -38.8, "socean_foa": 21.8, "socean_delta_odis": -103.3},
    2005: {"global_foa": 70.0, "global_delta_odis": -35.2, "socean_foa": 21.3, "socean_delta_odis": -100.2},
    2006: {"global_foa": 71.6, "global_delta_odis": -32.9, "socean_foa": 22.3, "socean_delta_odis": -97.3},
    2007: {"global_foa": 72.7, "global_delta_odis": -28.4, "socean_foa": 21.7, "socean_delta_odis": -92.4},
    2008: {"global_foa": 72.2, "global_delta_odis": -27.5, "socean_foa": 22.5, "socean_delta_odis": -91.5},
    2009: {"global_foa": 71.7, "global_delta_odis": -24.0, "socean_foa": 21.7, "socean_delta_odis": -88.9},
    2010: {"global_foa": 75.7, "global_delta_odis": -23.2, "socean_foa": 23.8, "socean_delta_odis": -87.8},
    2011: {"global_foa": 75.1, "global_delta_odis": -19.8, "socean_foa": 23.2, "socean_delta_odis": -84.9},
    2012: {"global_foa": 74.6, "global_delta_odis": -18.1, "socean_foa": 23.1, "socean_delta_odis": -83.9},
}


def compare_series(computed: dict, target: dict) -> dict:
    """Per-year {computed, target, pct_error} for every year in `target`."""
    result = {}
    for year, target_value in target.items():
        computed_value = computed[year]
        pct_error = 100 * (computed_value - target_value) / target_value
        result[year] = {
            "computed": computed_value,
            "target": target_value,
            "pct_error": pct_error,
        }
    return result


def load_ocediseq_annual(path, year: int) -> xr.DataArray:
    """Annual isoflux [PgC permil/yr] for one year from the paper's own
    gridded product (Ocediseq.nc), dims (lat, lon) -- a straight sum of
    that year's 12 monthly values, matching Table 1's own aggregator (see
    ocean14c.isoflux's module docstring for straight-sum vs. flux-weighted).
    """
    ds = xr.open_dataset(path)
    decimal_dates = ds["Decimal Date"].values
    year_indices = np.where((decimal_dates >= year) & (decimal_dates < year + 1))[0]
    annual = ds["Ocediseq"].isel(decdate=year_indices).sum(dim="decdate", skipna=True)
    return annual.rename({"Latitude": "lat", "Longitude": "lon"})


def spatial_correlation(a: xr.DataArray, b: xr.DataArray) -> float:
    """Pearson correlation between two gridded fields, over cells valid in
    both (NaN-safe, scale-invariant -- unaffected by e.g. a gamma rescaling
    of one of the two fields)."""
    both_valid = (~a.isnull()) & (~b.isnull())
    a_values = a.where(both_valid).values.ravel()
    b_values = b.where(both_valid).values.ravel()
    mask = ~np.isnan(a_values)
    return float(np.corrcoef(a_values[mask], b_values[mask])[0, 1])


def fit_gamma(foa_at_gamma, target: dict, bounds=(0.1, 0.6)) -> float:
    """Fit gamma by minimizing squared error of `foa_at_gamma(gamma)` (a
    dict[year] -> global Foa) against `target` (e.g. Table S2's Global_Foa).

    Pragmatic stand-in for the unpublished ERA-Interim-specific Sweeney-style
    recalibration -- see the plan's gas-transfer calibration discussion.
    """

    def objective(gamma):
        predicted = foa_at_gamma(gamma)
        return sum((predicted[year] - target[year]) ** 2 for year in target)

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    return result.x
