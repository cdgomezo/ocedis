"""Export every Delta_atm candidate as time series + 2012 maps, for the
comparison page.

    python3 scripts/export_delta_atm_comparison.py --start 2000 --end 2023

Writes results/delta_atm_comparison.json (scalar series) and
results/delta_atm_maps_2012.npz (fields). The plotting lives in
scripts/plot_delta_atm_comparison.py so this expensive step runs once.

The isoflux sweep re-runs the whole ocean pipeline for every (variant, year),
which reloads GLODAP and Takahashi each time -- a few minutes total. That is
cheaper than restructuring reconstruct_year, and this script is run rarely.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ocean14c import hua  # noqa: E402
from ocean14c.atm_stations import load_c14_station, load_drp_curl_event  # noqa: E402
from ocean14c.atmosphere import (  # noqa: E402
    NORTHERN_HEMISPHERE_WINDOWS, SOUTHERN_HEMISPHERE_WINDOWS,
    area_weighted_mean, delta_atm_zonal,
)
from ocean14c.pipeline import _load_atm_station_series, _monthly_hemispheric_series  # noqa: E402
from ocean14c.pipeline import reconstruct_year  # noqa: E402

GRAVEN = REPO_ROOT.parent / (
    "Delta_atm/Delta14co2-in-air_input4MIPs_atmosphericState_C4MIP_"
    "ImperialCollege-2-0_gz-NHTRSH_1850-2015.nc"
)

# Fixed order. A variant keeps its identity -- and therefore its colour --
# across every chart on the page.
VARIANTS = {
    "krige":        dict(geometry="krige"),
    "current":      dict(geometry="miller_band"),
    "basu_band":    dict(geometry="basu_band"),
    "hua_geom":     dict(geometry="hemispheric"),
    "hua_shape":    dict(geometry="hua", level="stations"),
    "hua_pub":      dict(geometry="hua", level="native"),
}
MAP_YEAR = 2012


def band_mean(lat, field, south, north):
    """cos(lat)-weighted mean over a latitude band of a (lat,) or (lat,lon)
    field."""
    arr = np.asarray(field, dtype=float)
    if arr.ndim == 2:
        arr = np.nanmean(arr, axis=1)
    m = (lat >= south) & (lat <= north)
    w = np.cos(np.radians(lat[m]))
    return float(np.nansum(w * arr[m]) / np.nansum(w * np.isfinite(arr[m])))


def graven_series():
    """Graven CMIP6 sectors as (year, global, nh, sh) -- annual, ends 2015."""
    import netCDF4 as nc

    if not GRAVEN.exists():
        return []
    with nc.Dataset(GRAVEN) as ds:
        t = 1850 + np.asarray(ds.variables["time"][:], dtype=float)
        v = np.asarray(ds.variables["Delta14co2_in_air"][:], dtype=float)
    # Sector area weights: |sin| of the band edges, NH 30..90, TR -30..30.
    w_nh = 1.0 - np.sin(np.radians(30.0))
    w_tr = 2.0 * np.sin(np.radians(30.0))
    out = []
    for i, year in enumerate(t):
        nh, tr, sh = v[i, 0], v[i, 1], v[i, 2]
        out.append({
            "year": int(round(year - 0.5)),
            "global": float((w_nh * nh + w_tr * tr + w_nh * sh) / (2 * w_nh + w_tr)),
            "nh": float(nh), "sh": float(sh),
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=2000)
    p.add_argument("--end", type=int, default=2023)
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    years = list(range(args.start, args.end + 1))
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    hua_series = hua.hemispheric_series()

    payload = {"years": years, "map_year": MAP_YEAR, "variants": {},
               "graven": graven_series(), "stations": {}}
    maps = {}

    for name, spec in VARIANTS.items():
        print(f"--- {name} ---", flush=True)
        rows = []
        for year in years:
            r = reconstruct_year(
                year, data_dir=args.data_dir, atm_variant=spec,
                krige_dir=str(REPO_ROOT / "output"), hua_series=hua_series,
            )
            da = r["delta_atm_monthly"].mean(dim="month")
            lat = np.asarray(da["lat"].values)
            prof = np.asarray(da.mean(dim="lon").values if "lon" in da.dims else da.values)
            rows.append({
                "year": year,
                "global": area_weighted_mean(lat, prof),
                "nh": band_mean(lat, prof, 30, 90),
                "sh": band_mean(lat, prof, -90, -30),
                "isoflux": float(r["global_isoflux"]),
                "delta_odis": float(r["global_delta_odis"]),
                "socean_delta_odis": float(r["socean_delta_odis"]),
            })
            if year == MAP_YEAR:
                full = r["delta_atm_monthly"].mean(dim="month")
                if "lon" not in full.dims:
                    full = full.broadcast_like(r["annual_isoflux"])
                maps[f"datm_{name}"] = np.asarray(full.transpose("lat", "lon").values,
                                                  dtype=np.float32)
                maps[f"isoflux_{name}"] = np.asarray(
                    r["annual_isoflux"].transpose("lat", "lon").values, dtype=np.float32)
                maps["lat"] = np.asarray(full["lat"].values, dtype=np.float32)
                maps["lon"] = np.asarray(full["lon"].values, dtype=np.float32)
            print(f"  {year} {rows[-1]['global']:7.2f}  {rows[-1]['isoflux']:9.1f}",
                  flush=True)
        payload["variants"][name] = rows

    # Station annual means, the observational reference the candidates target.
    nh_series, sh_series = _load_atm_station_series(
        Path(args.data_dir) / "raw" / "Delta_atm")
    for label, series, windows in (("nh", nh_series, NORTHERN_HEMISPHERE_WINDOWS),
                                   ("sh", sh_series, SOUTHERN_HEMISPHERE_WINDOWS)):
        payload["stations"][label] = [
            {"year": y,
             "value": float(np.mean(_monthly_hemispheric_series(series, windows, y)[0])),
             "station": _monthly_hemispheric_series(series, windows, y)[1]}
            for y in years
        ]

    # Hua's own hemispheric annual means, independent of our stations.
    payload["hua"] = [
        {"year": y,
         "nh": float(np.mean(hua.value_at(y + (np.arange(12) + 0.5) / 12, "NH",
                                          series=hua_series))),
         "sh": float(np.mean(hua.value_at(y + (np.arange(12) + 0.5) / 12, "SH",
                                          series=hua_series)))}
        for y in years
    ]

    (out_dir / "delta_atm_comparison.json").write_text(json.dumps(payload, indent=1))
    np.savez_compressed(out_dir / f"delta_atm_maps_{MAP_YEAR}.npz", **maps)
    print(f"\nwrote {out_dir/'delta_atm_comparison.json'}")
    print(f"wrote {out_dir}/delta_atm_maps_{MAP_YEAR}.npz")


if __name__ == "__main__":
    main()
