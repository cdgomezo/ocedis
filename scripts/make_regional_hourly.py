"""Write the ocean disequilibrium term on a regional, high-resolution,
hourly grid -- the EUROCOM 0.1 degree product.

    python scripts/make_regional_hourly.py --year 2024 \
        --resolution 0.1 --domain 33 73 -15 35 --tag eur010x010 \
        --atm-geometry krige --era5-dir data/processed/era5_hourly \
        --out-dir output/regional

Nothing here replaces the published path: scripts/validate_full_series.py and
ocean14c.pipeline still produce the global 1x1 monthly product that Table 1 /
Table S2 validate against, unchanged. This script is the regional counterpart
built on ocean14c.regional -- see that module for what hourly and 0.1 degree do
and do not buy (short version: the hourly structure is real and comes from the
wind; the 0.1 degree is an interpolation of 1x1 and 4x5 inputs).

It writes three things into --out-dir:

  ocedis_c14flux.<tag>.<year><mm>.nc   one per month, hourly, the C_D14C flux
      in umol m-2 s-1, stamped at the interval START (the preprocessor stamps
      the mid-point itself). Monthly files because a year of them is 7 GB in memory
      and 12 separate files concatenate cleanly along time.
  ocedis_delta_oce.<tag>.<year>.nc     monthly, permil, the surface-ocean
      Delta14C signature.
  ocedis_delta_atm.<tag>.<year>.nc     monthly, permil, the atmospheric
      Delta14C the disequilibrium was formed against.
  ocedis_consistency.<tag>.<year>.txt  the check described below.

THE CHECK. k = gamma * U^2 * (Sc/660)^-0.5 is linear in U^2, so if everything
but the wind is held monthly, the monthly mean of the hourly flux is the
monthly product exactly. That identity is what makes the disaggregation
trustworthy, so it is measured rather than asserted, against the same
reconstruction run monthly on the same grid. With --sst-source era5 the
hourly SST enters the Schmidt number and the solubility, both mildly
non-linear, and the identity holds to a fraction of a percent instead.

Foa IS NOT WRITTEN SEPARATELY and does not need to be: Delta_odis is constant
within a month, so the hourly one-way flux is recoverable exactly from the two
files that are written,

    Foa(t) [mol m-2 s-1] = c14flux(t) * 1e-6 * 1000 / (delta_oce - delta_atm)

taking Delta from the month t falls in.
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocean14c.regional import Grid, hourly_axis, reconstruct_regional  # noqa: E402

logger = logging.getLogger("make_regional_hourly")

CF_LAT = {"units": "degrees_north", "long_name": "latitude of grid cell centre",
          "standard_name": "latitude", "axis": "Y"}
CF_LON = {"units": "degrees_east", "long_name": "longitude of grid cell centre",
          "standard_name": "longitude", "axis": "X"}
CF_TIME = {"long_name": "mid of flux interval in UTC", "standard_name": "time",
           "axis": "T", "bounds": "time_bnds"}
TIME_ENCODING = {"units": "days since 1970-01-01", "calendar": "proleptic_gregorian",
                 "dtype": "float64", "_FillValue": None}


def _save(ds, path, compress=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {}
    for var in ds.data_vars:
        if np.issubdtype(ds[var].dtype, np.floating):
            encoding[var] = {"_FillValue": np.nan, "zlib": compress, "complevel": 4}
    for var in ds.coords:
        encoding[var] = {"_FillValue": None}
    for var in ("time", "time_bnds"):
        if var in ds.variables:
            encoding[var] = dict(TIME_ENCODING)
    if path.exists():
        path.unlink()
    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)
    logger.info(f"wrote {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return path


def _read_attrs(path):
    """Descriptive global attributes from a YAML/JSON file, as the preprocessor
    reads them. Empty keys are template placeholders and are dropped."""
    if path is None:
        return {}
    import yaml

    with open(path) as f:
        attrs = yaml.safe_load(f) or {}
    return {str(k): v for k, v in attrs.items() if v is not None and v != ""}


def _common_attrs(year, grid, result, args):
    prov = result["provenance"]
    return {
        "title": "Ocean 14C isotopic disequilibrium term over Europe",
        "source": (
            "Miller, Lehman & Lindsay (2025, GBC) ocean disequilibrium term, "
            "reconstructed from GLODAP (1x1) / Takahashi (4x5) / ERA5 / raw "
            "station Delta14C by the ocean14c package"
        ),
        "geospatial_lat_resolution": f"{grid.resolution:g} degree",
        "geospatial_lon_resolution": f"{grid.resolution:g} degree",
        "Conventions": "CF-1.8",
        "year": str(year),
        "delta_atm_geometry": args.atm_geometry,
        "regression_version": args.regression_version,
        "gas_transfer_gamma": args.gamma,
        "schmidt_polynomial": args.sc_poly,
        "atm_station_nh": prov["atm_station_nh"],
        "atm_station_sh": prov["atm_station_sh"],
        "atm_extrapolation_yr": float(prov["atm_extrapolation_yr"]),
        "history": (f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S} "
                    "written by Ocedis/scripts/make_regional_hourly.py"),
        "creation_date": f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d}",
    }


def write_flux_month(result, year, month, grid, out_dir, tag, args):
    """One month of hourly C_D14C flux, stamped at the interval start."""
    flux = result["c14flux"].sel(time=result["c14flux"]["time"].dt.month == month)
    starts = pd.DatetimeIndex(flux["time"].values)

    values = np.asarray(flux.transpose("time", "lat", "lon").values, dtype="float32")
    # Land and ice-free-ocean gaps are NaN all through the reconstruction. The
    # transport model must not see NaN, and zero is the physically right value:
    # no air-sea exchange where there is no sea.
    n_nan = int(np.isnan(values).sum())
    values = np.nan_to_num(values, nan=0.0)

    da = xr.DataArray(
        values, dims=("time", "lat", "lon"),
        # INTERVAL START, not the mid-point. The convention throughout is that
        # `time` is the start of each flux interval, so a consumer that wants
        # mid-points derives them from the bounds. Writing mid-points here
        # would shift the axis half a step late against everything else.
        coords={"time": starts, "lat": grid.lat.values, "lon": grid.lon.values},
        name="c14flux",
        attrs={
            "units": "umol m-2 s-1",
            "long_name": "Ocean isotopic disequilibrium 14C isoflux as C_D14C flux",
            "cell_methods": "time: mean area: mean",
            "comment": (
                "C_D14C = CO2 x Delta14C with Delta14C dimensionless "
                "(permil/1000), following Basu et al. (2016). Foa x Delta_odis, "
                "Foa the one-way ocean-to-atmosphere CO2 flux and Delta_odis = "
                "Delta_oce - Delta_atm. SIGN FOLLOWS Delta_odis: the global "
                "historical term is negative, but by 2024 atmospheric Delta14CO2 "
                "has fallen below the surface-ocean value over this domain, so "
                "the flux here is positive. Land cells are 0."
            ),
        },
    )
    ds = da.to_dataset()
    ds["lat"].attrs, ds["lon"].attrs, ds["time"].attrs = CF_LAT, CF_LON, dict(CF_TIME)
    ds["time_bnds"] = xr.DataArray(
        np.stack([starts.values, (starts + pd.Timedelta(hours=1)).values], axis=-1),
        dims=("time", "nbnds"), coords={"time": ds["time"]})
    ds["time"].attrs["long_name"] = "start of flux interval in UTC"
    ds.attrs = _common_attrs(year, grid, result, args)
    ds.attrs.update(_read_attrs(args.attrs))
    ds.attrs["frequency"] = "hourly"
    ds.attrs["land_cells"] = f"{n_nan} NaN cells this month written as 0.0"
    ds.attrs["sst_source"] = result.get("sst_source", "climatology")

    return _save(ds, Path(out_dir) / f"ocedis_c14flux.{tag}.{year}{month:02d}.nc")


def write_permil(result, key, year, grid, out_dir, tag, args, long_name, comment):
    """A monthly permil field (Delta_oce or Delta_atm) on the target grid.

    Stamped at the month START, like the flux months. The convention
    throughout is that `time` is the start of the interval, so a consumer
    that wants mid-points derives them from the bounds; writing mid-points
    here would shift the axis half a month late.
    """
    da = result[key].transpose("month", "lat", "lon")
    months = np.asarray(da["month"].values, dtype=int)
    starts = pd.to_datetime([f"{year}-{m:02d}-01" for m in months])
    ends = starts + pd.offsets.MonthBegin(1)

    out = xr.DataArray(
        np.asarray(da.values, dtype="float32"), dims=("time", "lat", "lon"),
        coords={"time": starts, "lat": grid.lat.values, "lon": grid.lon.values},
        name=key,
        attrs={"units": "permil", "long_name": long_name,
               "cell_methods": "time: mean area: mean", "comment": comment},
    ).to_dataset()
    out["lat"].attrs, out["lon"].attrs, out["time"].attrs = CF_LAT, CF_LON, dict(CF_TIME)
    out["time"].attrs["long_name"] = "start of month in UTC"
    out["time_bnds"] = xr.DataArray(
        np.stack([starts.values, ends.values], axis=-1),
        dims=("time", "nbnds"), coords={"time": out["time"]})
    out.attrs = _common_attrs(year, grid, result, args)
    out.attrs.update(_read_attrs(args.attrs))
    out.attrs["frequency"] = "monthly"
    out.attrs["comment"] = (
        "Delta14C signature field, NOT a flux: intensive, so it has no total and "
        "its country aggregate is an area-weighted mean. Monthly because that is "
        "its native resolution -- the ocean signature moves on a decadal trend "
        "and the atmospheric one on a seasonal cycle, and neither has "
        "sub-monthly structure to disaggregate."
    )
    return _save(out, Path(out_dir) / f"ocedis_{key}.{tag}.{year}.nc")


def consistency_check(result, year, grid, args, out_path):
    """Decompose the hourly product against the monthly reconstruction.

    The first version of this check reported ONE number -- the monthly mean of
    the hourly flux against the monthly product -- and it came out at 3.8% on
    the domain mean, which I wrongly attributed to the mild non-linearity of
    hourly SST. It is not that. Three things differ between the two, and only
    the first is the disaggregation:

      A  hourly -> monthly aggregation, same winds and same SST. This is the
         identity the hourly product rests on and it is EXACT (k is linear in
         U^2), so anything but zero here is a bug.
      B  SST source: ERA5 hourly against the Takahashi climatology. This is the
         whole of the difference. The climatology is 4x5 degrees, sampled
         nearest-neighbour, and over a domain that is mostly marginal seas it
         is far colder than the real 2024 field -- measured 6.76 degC against
         10.66 degC domain-mean in January.
      C  wind product: the shipped global monthly <U^2> (1 degree, 6-hourly)
         against the monthly mean of the hourly 0.25 degree field. Small.
    """
    from ocean14c.foa import compute_foa
    from ocean14c.isoflux import to_umol_flux
    from ocean14c.regional import _monthly_u2, load_era5_hourly

    w = np.cos(np.deg2rad(grid.lat.values))[:, None]

    def domain_mean(x):
        v = np.asarray(x)
        m = np.isfinite(v)
        return float(np.nansum(v * w * m) / np.nansum(w * m))

    era5_paths = sorted(Path(args.era5_dir).glob(args.era5_glob.format(year=year)))
    delta_odis = result["delta_oce"] - result["delta_atm"]
    sst_clim = result["sst_clim"]
    pco2 = result["pco2_ocean"]
    u2_prod = _monthly_u2(year, grid, args.data_dir)

    rows = []
    for path in era5_paths:
        u2h, ssth = load_era5_hourly([path], grid)
        u2h = u2h.compute(); ssth = ssth.compute()
        month = int(pd.DatetimeIndex(u2h["time"].values)[0].month)
        odis = delta_odis.sel(month=month)
        pc = pco2.sel(month=month)
        sc = sst_clim.sel(month=month)

        hourly_clim = to_umol_flux(compute_foa(u2h, sc, pc, gamma=args.gamma,
                                               sc_poly=args.sc_poly), odis).mean("time")
        monthly_same = to_umol_flux(compute_foa(u2h.mean("time"), sc, pc, gamma=args.gamma,
                                                sc_poly=args.sc_poly), odis)
        hourly_era5 = to_umol_flux(compute_foa(u2h, ssth, pc, gamma=args.gamma,
                                               sc_poly=args.sc_poly), odis).mean("time")
        monthly_prod = to_umol_flux(compute_foa(u2_prod.sel(month=month), sc, pc,
                                                gamma=args.gamma, sc_poly=args.sc_poly), odis)

        a, b0 = domain_mean(hourly_clim), domain_mean(monthly_same)
        rows.append(dict(
            month=month,
            A=abs(a - b0) / abs(b0),
            B=(domain_mean(hourly_era5) - a) / abs(a),
            C=(domain_mean(monthly_prod) - b0) / abs(b0),
            sst_clim=domain_mean(sc), sst_era5=domain_mean(ssth.mean("time")),
            flux=domain_mean(hourly_era5),
        ))

    df = pd.DataFrame(rows).sort_values("month")
    lines = [
        "Hourly-vs-monthly decomposition, ocean disequilibrium C_D14C flux",
        f"  year {year}, grid {grid.describe()}, sst_source = {result.get('sst_source')}",
        "",
        "  A  hourly -> monthly aggregation (same winds, same SST): the identity the",
        "     hourly product rests on, k being linear in U^2. Must be ~0.",
        "  B  SST source, ERA5 hourly vs the Takahashi 4x5 climatology, winds equal.",
        "  C  wind product, shipped 1 deg 6-hourly <U^2> vs monthly mean of hourly 0.25 deg.",
        "",
        f"  {'month':>5s} {'A [rel]':>10s} {'B [%]':>8s} {'C [%]':>8s} "
        f"{'SST clim':>9s} {'SST era5':>9s} {'flux':>11s}",
    ]
    for _, r_ in df.iterrows():
        lines.append(f"  {int(r_.month):5d} {r_.A:10.2e} {100*r_.B:+8.2f} {100*r_.C:+8.2f} "
                     f"{r_.sst_clim:9.2f} {r_.sst_era5:9.2f} {r_.flux:11.4e}")
    lines += [
        "",
        f"  A: max over months {df.A.max():.2e}  -- exact aggregation, as designed",
        f"  B: mean {100*df.B.mean():+.2f} %, range {100*df.B.min():+.2f} .. {100*df.B.max():+.2f} %",
        f"  C: mean {100*df.C.mean():+.2f} %, range {100*df.C.min():+.2f} .. {100*df.C.max():+.2f} %",
        f"  domain-mean SST: climatology {df.sst_clim.mean():.2f} degC, ERA5 {df.sst_era5.mean():.2f} degC",
        "",
        "  The shipped product uses ERA5 SST: at 0.1 degree over marginal seas the",
        "  4x5 climatology is the wrong field, and B is the price of being right",
        "  rather than an error. Re-run with --sst-source climatology to reproduce",
        "  the global monthly product exactly instead.",
    ]
    txt = "\n".join(lines)
    Path(out_path).write_text(txt + "\n")
    logger.info("\n" + txt)
    df.to_csv(str(out_path).replace(".txt", ".csv"), index=False)
    return float(df.A.max())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--resolution", type=float, default=0.1)
    p.add_argument("--domain", type=float, nargs=4, required=True,
                   metavar=("LAT0", "LAT1", "LON0", "LON1"))
    p.add_argument("--tag", default="eur010x010")
    p.add_argument("--era5-dir", default=None,
                   help="directory of the hourly ERA5 files. Omitted: monthly output.")
    p.add_argument("--era5-glob", default="era5_hourly_*_{year}*.nc")
    p.add_argument("--out-dir", default="output/regional")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--krige-dir", default="output")
    p.add_argument("--atm-geometry", default="krige",
                   choices=["krige", "miller_band", "basu_band", "hemispheric", "hua"])
    p.add_argument("--sst-source", default="era5", choices=["era5", "climatology"])
    p.add_argument("--regression-version", default="original")
    p.add_argument("--gamma", type=float, default=0.31)
    p.add_argument("--sc-poly", default="1992", choices=["1992", "2014"])
    p.add_argument("--months", type=int, nargs="*", default=None)
    p.add_argument("--attrs", default=None, metavar="FILE",
                   help="YAML file of descriptive global attributes "
                        "(institution, creator, references, summary, ...), "
                        "merged into every file written here.")
    p.add_argument("--skip-check", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    lat0, lat1, lon0, lon1 = args.domain
    grid = Grid(args.resolution, lat0, lat1, lon0, lon1)
    logger.info(f"target grid: {grid.describe()}")

    era5_paths = None
    if args.era5_dir:
        pattern = args.era5_glob.format(year=args.year)
        era5_paths = sorted(Path(args.era5_dir).glob(pattern))
        if not era5_paths:
            raise SystemExit(f"no ERA5 files matching {args.era5_dir}/{pattern}")
        logger.info(f"{len(era5_paths)} hourly ERA5 files")

    result = reconstruct_regional(
        args.year, grid, era5_paths=era5_paths, data_dir=args.data_dir,
        krige_dir=args.krige_dir, gamma=args.gamma, sc_poly=args.sc_poly,
        regression_version=args.regression_version,
        atm_variant={"geometry": args.atm_geometry},
        sst_source=args.sst_source, months=args.months,
    )

    out_dir = Path(args.out_dir)
    months = args.months or list(range(1, 13))

    if result["freq"] == "hourly":
        for month in months:
            write_flux_month(result, args.year, month, grid, out_dir, args.tag, args)
    else:
        logger.warning("no ERA5 directory given: writing the MONTHLY field only")

    write_permil(result, "delta_oce", args.year, grid, out_dir, args.tag, args,
                 "surface ocean Delta14C",
                 "Delta14C of surface-ocean DIC, projected from the GLODAP 1995 "
                 "field by the Miller/Lindsay regression. Capital Delta per mil "
                 "(Stuiver & Polach 1977).")
    write_permil(result, "delta_atm", args.year, grid, out_dir, args.tag, args,
                 "atmospheric Delta14CO2",
                 f"Delta14C of atmospheric CO2, geometry '{args.atm_geometry}'. "
                 "This is the field the ocean disequilibrium was formed against; "
                 "Delta_odis = delta_oce - delta_atm.")

    if result["freq"] == "hourly" and not args.skip_check:
        consistency_check(result, args.year, grid, args,
                          out_dir / f"ocedis_consistency.{args.tag}.{args.year}.txt")


if __name__ == "__main__":
    main()
