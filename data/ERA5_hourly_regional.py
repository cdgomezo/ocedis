"""Fetch HOURLY ERA5 wind and SST for one year over one regional domain.

    python data/ERA5_hourly_regional.py --year 2024 \
        --domain 33 73 -15 35 --out-dir data/processed/era5_hourly

ocean14c.atmosphere_forcing builds the GLOBAL, MONTHLY <U^2> the published
reconstruction runs on (1x1, 6-hourly sampled, one file per year). This script
is its regional, hourly counterpart: the gas-transfer velocity goes as U^2, so
an hourly flux needs hourly wind, and a monthly mean cannot be unpicked into
one.

What it writes, per month, into --out-dir:

    era5_hourly_<domain>_<year><mm>.nc
        u2  (time, latitude, longitude)  [m2 s-2]  u10^2 + v10^2, instantaneous
        sst (time, latitude, longitude)  [degC]    sea surface temperature

on ERA5's native 0.25 degree grid, cropped to the domain. The raw download is
deleted once reduced -- u10/v10 are only ever needed as
their second moment, and keeping a year of them costs ~3 GB for nothing.

MONTH BY MONTH on purpose: one request per year would sit in the CDS queue for
hours and lose everything if it failed, and the reduction needs the whole file
in memory. Each month is ~290 MB raw and a few seconds of work.

NOTE the time convention. ERA5 single-level wind and SST are INSTANTANEOUS
values valid at the stamped time, not interval means, so `u2` at 00:00 is the
instantaneous square, and the hourly flux built from it is treated as the mean
over the hour beginning at that stamp (see scripts/make_regional_hourly.py,
which stamps the flux interval mid-point, 00:30, matching the CTE-HR files).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr

HOURS = [f"{h:02d}:00" for h in range(24)]
DAYS = [f"{d:02d}" for d in range(1, 32)]


def fetch_month(client, year: int, month: int, area, out_path: Path, grid: float = 0.25):
    """Download one month of hourly u10/v10/SST and reduce it to u2 + sst.

    area : (north, west, south, east) in degrees, the CDS convention.
    """
    raw = out_path.with_suffix(".raw.nc")

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "sea_surface_temperature",
            ],
            "year": str(year),
            "month": f"{month:02d}",
            "day": DAYS,
            "time": HOURS,
            "area": list(area),
            "grid": [grid, grid],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(raw),
    )

    with xr.open_dataset(raw) as ds:
        # ERA5 renamed `time` to `valid_time` in the 2024 CDS-Beta rewrite, and
        # the two coexist in the wild -- accept either rather than guessing.
        tname = "valid_time" if "valid_time" in ds.dims else "time"
        u2 = (ds["u10"] ** 2 + ds["v10"] ** 2).astype("float32")
        u2.attrs = {"units": "m2 s-2", "long_name": "square of 10 m wind speed"}
        # SST is in K and is NaN over land; kept as NaN, since a zero there
        # would be -273 degC and would poison the Schmidt number.
        sst = (ds["sst"] - 273.15).astype("float32")
        sst.attrs = {"units": "degC", "long_name": "sea surface temperature"}
        out = xr.Dataset({"u2": u2, "sst": sst})
        if tname != "time":
            out = out.rename({tname: "time"})
        out.attrs = {
            "source": "ERA5 reanalysis, hourly single levels",
            "note": (
                "u2 = u10^2 + v10^2 from instantaneous hourly winds; the "
                "gas-transfer velocity needs <U^2>, never <U>^2"
            ),
        }
        encoding = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
        out.to_netcdf(out_path, encoding=encoding)

    os.remove(raw)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--domain", type=float, nargs=4, required=True,
                   metavar=("LAT0", "LAT1", "LON0", "LON1"),
                   help="south north west east, in degrees")
    p.add_argument("--out-dir", default="data/processed/era5_hourly")
    p.add_argument("--grid", type=float, default=0.25,
                   help="ERA5 output grid spacing in degrees (default: 0.25, native)")
    p.add_argument("--months", type=int, nargs="*", default=list(range(1, 13)))
    p.add_argument("--tag", default=None,
                   help="domain tag used in the filename (default: from --domain)")
    args = p.parse_args(argv)

    lat0, lat1, lon0, lon1 = args.domain
    # A margin of one ERA5 cell, so interpolation onto the target grid never
    # needs to extrapolate at the domain edge.
    area = (min(lat1 + args.grid, 90.0), max(lon0 - args.grid, -180.0),
            max(lat0 - args.grid, -90.0), min(lon1 + args.grid, 180.0))
    tag = args.tag or f"{lat0:g}_{lat1:g}_{lon0:g}_{lon1:g}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import cdsapi

    client = cdsapi.Client()

    for month in args.months:
        out_path = out_dir / f"era5_hourly_{tag}_{args.year}{month:02d}.nc"
        if out_path.exists():
            print(f"{out_path.name} exists, skipping")
            continue
        print(f"fetching {args.year}-{month:02d} over N{area[0]} W{area[1]} "
              f"S{area[2]} E{area[3]} ...", flush=True)
        fetch_month(client, args.year, month, area, out_path, grid=args.grid)
        size = out_path.stat().st_size / 1e6
        print(f"  -> {out_path} ({size:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
