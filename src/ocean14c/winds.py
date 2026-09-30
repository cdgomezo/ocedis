"""Load ERA5 <U^2> (mean of squared wind speed) produced by
ocean14c.atmosphere_forcing (data/processed/era5_u2/u2_<year>.nc).

Each file holds u10**2 + v10**2 resampled to monthly means, with longitude
in ERA5's native 0-360 convention. This module standardizes that onto
(month, lat, lon) with longitude in -180..180, matching GLODAP.
"""

import xarray as xr

from ocean14c.atmosphere_forcing import _time_coord_name


def load_u2(path) -> xr.DataArray:
    """<U^2> for one year's ERA5 output file, dims (month, lat, lon)."""
    ds = xr.open_dataset(path)
    (var_name,) = ds.data_vars
    field = ds[var_name]

    field = field.rename({"latitude": "lat", "longitude": "lon"})
    field = field.assign_coords(lon=((field["lon"] + 180) % 360) - 180).sortby("lon")

    time_dim = _time_coord_name(ds)
    field = field.rename({time_dim: "month"})
    field = field.assign_coords(month=field["month"].dt.month.values)

    return field.rename("u2")
