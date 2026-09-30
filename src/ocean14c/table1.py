"""Table 1-matched variant of the TM5-facing ocean disequilibrium flux.

WHAT THIS IS
------------
`pipeline.reconstruct_year` rebuilds FoaDelta_odis from raw data. Against
Miller, Lehman & Lindsay (2025) it gets the *pattern* right (0.90-0.93
spatial correlation against their own gridded product, every year) but sits
~8-17% low in *magnitude*, stably across 2000-2012 (docs/SCIENCE.md 8).

Some downstream uses -- an atmospheric 14CO2 simulation whose Delta14C trend
is driven by the annual global magnitude of this term -- want a product that
carries the paper's published numbers rather than this project's independent
ones. This module builds that alternative by rescaling each year's field so
its annual global isoflux equals Table 1 exactly.

It is an AMPLITUDE CORRECTION, not a second reconstruction: one scalar per
year multiplies the whole field, so the spatial pattern, the seasonal cycle
and the sign are those of the reconstruction, unchanged. That is the same
kind of operation as changing gamma, which is a uniform multiplicative
constant (gas_exchange.transfer_velocity) -- so the Table 1 variant is
exactly what a per-year gamma calibrated on Table 1 would have produced,
and it leaves the spatial correlation against Ocediseq.nc untouched.

WHAT IT IS NOT
--------------
It is not evidence that the reconstruction was wrong, and it does not close
the gap documented in docs/SCIENCE.md 8-8.2 -- it papers over it by fiat.
Note in particular 8.2: correcting the independently-corroborated
northern-North-Pacific bias moves the total *away* from Table 1 (-11.8% ->
-16.7%), so part of the agreement being enforced here may be a shared bias.
Use `reconstruction` when the question is "what does the raw data say", and
`table1` when the question is "what would TM5 do with the published term".

OUT-OF-TABLE YEARS
------------------
Table 1 stops at 2012; the flux archive runs to 2025. Those years are scaled
by the mean of the in-window factors. A linear fit of factor against year
over 2000-2012 is not significant under either curve fit (simplified:
+0.0011/yr, p=0.32; ccgcrv: -0.0032/yr, p=0.09), so there is nothing to
extrapolate and a constant is the honest choice. The constant also lands
within 0.3% of the factor that would match Table 1's SUM row (simplified
1.1349 vs 1.1338; ccgcrv 1.1271 vs 1.1299), so it is not quietly a different
calibration from the sum-matched one.
"""

import numpy as np

from ocean14c.validate import TABLE_1_FOA_DELTA_ODIS

# Per-year factors under the reference settings below. Used ONLY as the
# fallback for out-of-table years when the current run reconstructed no
# in-table year to derive a mean from; whenever an in-table year IS available
# its factor is recomputed live, so a change in gamma or in the regression
# version is picked up automatically.
#
# Keyed by curve fit because the two differ by up to 4.6% in a single year
# (2011: 1.1598 vs 1.1093) -- more than the spread within either series. Both
# sets are reproducible from a clean tree:
#
#   python3 scripts/validate_full_series.py --start-year 2000 --end-year 2012 \
#       --gamma 0.21469736017350816 [--curve-fit ccgcrv] --flux-variant table1
#
# ccgcrv is the one that produced results/full_series_results.csv and the
# shipped output/flux_c14.ocedis.<year>.nc, and is required past 2023
# (docs/SCIENCE.md 9); `simplified` is the script's default.
REFERENCE_SETTINGS = {
    "gamma": 0.21469736017350816,  # fit against Table S2's Global_Foa, docs/SCIENCE.md 6.3
    "regression_version": "original",
    "sc_poly": "1992",
}
REFERENCE_SCALE_FACTORS = {
    "simplified": {
        2000: 1.1320, 2001: 1.1331, 2002: 1.1374, 2003: 1.1303, 2004: 1.1356,
        2005: 1.1365, 2006: 1.1423, 2007: 1.1203, 2008: 1.1143, 2009: 1.1193,
        2010: 1.1334, 2011: 1.1598, 2012: 1.1594,
    },
    "ccgcrv": {
        2000: 1.1223, 2001: 1.1248, 2002: 1.1620, 2003: 1.1634, 2004: 1.1267,
        2005: 1.1668, 2006: 1.1169, 2007: 1.1015, 2008: 1.1467, 2009: 1.0835,
        2010: 1.1143, 2011: 1.1093, 2012: 1.1137,
    },
}
REFERENCE_MEAN_SCALE = {
    fit: float(np.mean(list(factors.values())))
    for fit, factors in REFERENCE_SCALE_FACTORS.items()
}

DEFAULT_CURVE_FIT = "simplified"

TERM_RECONSTRUCTION = "ocedis"
TERM_TABLE1 = "ocedis_table1"


def reference_mean_scale(curve_fit=DEFAULT_CURVE_FIT):
    """Frozen fallback factor for `curve_fit`, for a run that covers no year
    Table 1 covers. Unknown curve fits fall back to the default rather than
    raising: a new curve fit is a reason to warn, not to lose the run."""
    return REFERENCE_MEAN_SCALE.get(curve_fit, REFERENCE_MEAN_SCALE[DEFAULT_CURVE_FIT])


def scale_factor(year, global_isoflux):
    """Factor that takes `global_isoflux` [PgC permil/yr] to Table 1's value.

    Returns None for a year Table 1 does not cover -- the caller decides what
    to do with those (see `scale_factors`).

    Raises ValueError if the reconstructed value is zero or has the opposite
    sign to Table 1's: a negative or infinite factor would flip the field
    inside out rather than rescale it, and silently shipping that to TM5 is
    the one failure mode here that would not be visible in a map.
    """
    target = TABLE_1_FOA_DELTA_ODIS.get(year)
    if target is None:
        return None
    value = float(global_isoflux)
    if not np.isfinite(value) or value == 0.0:
        raise ValueError(
            f"{year}: reconstructed global isoflux is {value!r}; cannot scale "
            f"it onto Table 1's {target}"
        )
    factor = target / value
    if factor <= 0.0:
        raise ValueError(
            f"{year}: reconstructed global isoflux {value:.1f} has the opposite "
            f"sign to Table 1's {target:.1f} (factor {factor:.3f}). This is an "
            f"amplitude correction, not a sign correction -- something is wrong "
            f"upstream."
        )
    return factor


def scale_factors(global_isoflux_by_year, fallback=None,
                  curve_fit=DEFAULT_CURVE_FIT):
    """{year: factor} for every year in `global_isoflux_by_year`.

    In-table years get their own exact factor. Out-of-table years get
    `fallback`, which defaults to the mean of the in-table factors present in
    this same call -- so a run covering 2000-2025 is internally consistent,
    and a run covering only 2013-2025 falls back to this curve fit's frozen
    REFERENCE_MEAN_SCALE, with the caller warned via `uses_reference_fallback`.
    """
    factors = {
        year: scale_factor(year, isoflux)
        for year, isoflux in global_isoflux_by_year.items()
    }

    in_table = [f for f in factors.values() if f is not None]
    if fallback is None:
        fallback = (float(np.mean(in_table)) if in_table
                    else reference_mean_scale(curve_fit))
    return {year: (f if f is not None else fallback) for year, f in factors.items()}


def uses_reference_fallback(global_isoflux_by_year):
    """True if any year would be scaled by the frozen REFERENCE_MEAN_SCALE
    rather than by factors derived in this run -- i.e. the run covers no year
    Table 1 covers, and the fallback's validity depends on the current
    settings matching REFERENCE_SETTINGS."""
    years = set(global_isoflux_by_year)
    return bool(years) and not (years & set(TABLE_1_FOA_DELTA_ODIS))


def scale_flux(c14flux_monthly, factor):
    """Rescale a monthly c14flux field, preserving coords, dims and NaN mask."""
    return c14flux_monthly * float(factor)


def describe(factors):
    """One-line-per-year provenance string for logs and the NetCDF header."""
    lines = []
    for year in sorted(factors):
        target = TABLE_1_FOA_DELTA_ODIS.get(year)
        origin = f"Table 1 target {target:.1f} PgC permil/yr" if target is not None \
            else "mean of in-window factors (Table 1 stops at 2012)"
        lines.append(f"{year}: x{factors[year]:.4f}  [{origin}]")
    return "\n".join(lines)
