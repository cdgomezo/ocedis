"""Project surface ocean Delta14C-DIC forward in time from the 1995 GLODAP
field (CLAUDE.md Sec 3.3).

Delta_oce(lat, lon, t) = Delta_oce_1995(lat, lon) + rate(lat, lon) * (t - 1995.0)
"""

import xarray as xr

from ocean14c.regions import is_cold_population, northern_north_pacific_mask
from ocean14c.regression import rate_of_change

REFERENCE_YEAR = 1995.0


def project_delta_oce(
    delta_1995: xr.DataArray,
    year: float,
    version: str = "original",
    threshold: float = 20.0,
) -> xr.DataArray:
    """Delta_oce(t) for every grid cell in `delta_1995`.

    The northern North Pacific mask is always computed and passed through;
    `rate_of_change` ignores it unless `version` defines an "npac"
    population, so the two-population versions are bit-for-bit unaffected.
    """
    is_cold = is_cold_population(delta_1995, threshold=threshold)
    is_npac = northern_north_pacific_mask(delta_1995)
    # Wrapped rather than passed positionally: rate_of_change's third
    # positional parameter is `version`, not `is_npac`.
    rate = xr.apply_ufunc(
        lambda d, cold, npac: rate_of_change(d, cold, version=version, is_npac=npac),
        delta_1995,
        is_cold,
        is_npac,
    )
    return delta_1995 + rate * (year - REFERENCE_YEAR)
