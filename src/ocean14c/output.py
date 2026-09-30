"""NetCDF writer for the gridded ocean disequilibrium flux product.

One convention, so that several C_D14C flux terms can be combined:

    filename : flux_c14.<term>.<year>.nc
    variable : c14flux, dims (time, lat, lon), units umol m-2 s-1

There is no vertical dimension, since a transport model distributes surface
fluxes itself.
For the ocean term that is trivially correct in any case: air-sea exchange
is a surface flux by construction.
"""

import datetime
import os

import netCDF4 as nc
import numpy as np

def month_start_days(year):
    """Day-of-year offsets of the 12 month starts relative to {year}-01-01.

    Monthly fluxes are stamped at the START of the month, the convention used
    throughout, so that files can be combined on a common time axis without
    alignment surprises.

    Computed rather than tabulated so leap years are correct: 1 March is day
    59 in a common year and day 60 in a leap year.
    """
    jan1 = datetime.date(year, 1, 1)
    return [float((datetime.date(year, m, 1) - jan1).days) for m in range(1, 13)]


def write_year(year, flux_monthly, out_dir, compress=True, fill_land_with_zero=True,
               term="ocedis", scaling=None):
    """Write one year of ocean C_D14C flux to flux_c14.{term}.{year}.nc.

    year         : calendar year (int)
    flux_monthly : xr.DataArray [umol C_D14C m^-2 s^-1] with dims
                   (month, lat, lon) -- as returned by
                   ocean14c.pipeline.reconstruct_year()["c14flux_monthly"]
    out_dir      : directory to write into (created if absent)

    term         : the `<term>` field of the filename. "ocedis" (default) is
        the reconstruction; "ocedis_table1" is the Table 1-matched variant
        (ocean14c.table1). Distinct names on purpose: the two products are
        written side by side into the same directory, and a consumer selects
        between them by this name.
    scaling      : optional provenance string recorded in the `scaling` global
        attribute, e.g. ocean14c.table1.describe()'s line for this year. Left
        unset the attribute says the field is unscaled, so a file is always
        self-describing about which variant it holds.

    fill_land_with_zero : land and other invalid cells are NaN throughout the
        reconstruction (no GLODAP/Takahashi coverage). TM5 emission fields
        must not contain NaN -- a single one would poison the tracer -- so
        they are written as 0.0 by default, which is physically right: no
        air-sea exchange over land.

    Returns the path written.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"flux_c14.{term}.{year}.nc")

    data = flux_monthly.transpose("month", "lat", "lon")
    lat = np.asarray(data["lat"].values, dtype="f8")
    lon = np.asarray(data["lon"].values, dtype="f8")
    values = np.asarray(data.values, dtype="f8")

    n_land = int(np.isnan(values).sum())
    if fill_land_with_zero:
        values = np.nan_to_num(values, nan=0.0)

    nmonth, nlat, nlon = values.shape
    if nmonth != 12:
        raise ValueError(f"expected 12 months, got {nmonth}")

    zlib = compress
    clevel = 4 if compress else 0

    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", 12)
        ds.createDimension("lat", nlat)
        ds.createDimension("lon", nlon)

        t = ds.createVariable("time", "f8", ("time",))
        t.units = f"days since {year}-01-01 00:00:00"
        t.calendar = "standard"
        t.long_name = "time"
        t.standard_name = "time"
        t.axis = "T"
        t[:] = np.array(month_start_days(year), dtype="f8")

        v = ds.createVariable("lat", "f4", ("lat",))
        v.units = "degrees_north"
        v.long_name = "latitude"
        v.standard_name = "latitude"
        v[:] = lat.astype("f4")

        v = ds.createVariable("lon", "f4", ("lon",))
        v.units = "degrees_east"
        v.long_name = "longitude"
        v.standard_name = "longitude"
        v[:] = lon.astype("f4")

        c14 = ds.createVariable(
            "c14flux", "f4", ("time", "lat", "lon"), zlib=zlib, complevel=clevel
        )
        c14.units = "umol m-2 s-1"
        c14.long_name = "Ocean isotopic disequilibrium 14C isoflux as C_D14C flux"
        c14.comment = (
            "C_D14C = CO2 x Delta14C with Delta14C dimensionless (permil/1000), "
            "following Basu et al. (2016). Computed as Foa x Delta_odis, where "
            "Foa is the one-way ocean-to-atmosphere CO2 flux and Delta_odis = "
            "Delta_oce - Delta_atm. Negative over essentially the whole ocean. "
            "Land cells are 0."
        )
        c14[:] = values.astype("f4")

        ds.title = "Ocean 14C disequilibrium isoflux as C_D14C flux for TM5"
        ds.year = str(year)
        ds.method = (
            "Miller, Lehman & Lindsay (2025, GBC) ocean disequilibrium term, "
            "reconstructed from GLODAP / Takahashi / ERA5 / raw station Delta14C"
        )
        ds.term = term
        ds.scaling = scaling if scaling is not None else (
            "none - unscaled reconstruction"
        )
        ds.vertical_grid = "none - surface flux"
        ds.land_cells = f"{n_land} NaN cells per year written as 0.0"
        ds.created = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return path
