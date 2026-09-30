"""Build the gridded atmospheric Delta14C field from the station network.

    python3 scripts/build_delta_atm_grid.py
    python3 scripts/build_delta_atm_grid.py --start 1990 --end 2025 --res 1.0

Writes output/d14c_atm.<year>.nc (monthly, permil, with an uncertainty field)
plus output/d14c_atm_timeseries.nc, the scalar Delta14C(year, month) series in
the format TM5's `atmospheric.D14C.timeseries.output` rc-key reads today.

See docs/SCIENCE.md Sec 10 for the method and its limits -- in particular,
cross-validation shows the network constrains a linear interhemispheric
gradient and little else, so the field is close to zonally uniform by
construction rather than by assumption.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ocean14c.atmfield import (  # noqa: E402
    build, global_mean_series, write_timeseries, write_year,
)
from ocean14c.background import report_table  # noqa: E402
from ocean14c.stations import registry_table  # noqa: E402

DELTA_ATM_DIR = REPO_ROOT / "data/raw/Delta_atm"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=1990)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--res", type=float, default=1.0, help="grid resolution [deg]")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "output"))
    p.add_argument("--include-regional", action="store_true",
                   help="retain fossil-affected sites (downweighted) instead of "
                        "excluding them")
    p.add_argument("--data-dir", default=str(DELTA_ATM_DIR))
    p.add_argument("--max-extrapolation-yr", type=float, default=0.0,
                   help="admit stations whose record ends up to this many years "
                        "before the target month, extrapolating the curve. "
                        "2.0 is needed to reach 2025, where only JFJ still "
                        "reports; their error is inflated accordingly.")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    out_dir = Path(args.out_dir)

    print(f"Building gridded Delta14C, {args.start}-{args.end}, {args.res} deg")
    result = build(
        args.data_dir, args.start, args.end,
        res_deg=args.res, include_regional=args.include_regional,
        max_extrapolation_yr=args.max_extrapolation_yr,
    )

    print()
    print("Station registry")
    print(registry_table(result["stations"]))
    print()
    print("Background filter")
    print(report_table(result["reports"]))

    n = result["n_stations"]
    written = 0
    for year in range(args.start, args.end + 1):
        sel = result["years"] == year
        if not np.isfinite(result["field"][sel]).any():
            print(f"  {year}: SKIPPED (fewer than the minimum stations all year)")
            continue
        write_year(result, year, out_dir)
        written += 1

    ts = write_timeseries(result, str(out_dir / "d14c_atm_timeseries.nc"))

    g = global_mean_series(result)
    print()
    print(f"wrote {written} annual files to {out_dir}")
    print(f"wrote {ts}")
    print(f"stations per month: {n.min()}-{n.max()} (mean {n.mean():.1f})")
    print(f"global mean Delta14C: {np.nanmax(g):+.1f} permil "
          f"({args.start}) -> {np.nanmin(g):+.1f} permil ({args.end})")


if __name__ == "__main__":
    main()
