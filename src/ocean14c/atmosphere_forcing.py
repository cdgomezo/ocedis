"""Download and process ERA5 10m winds into monthly <U^2> for the
gas-exchange pipeline (CLAUDE.md Sec 4.1, 7; consumed by gas_exchange.py /
foa.py via ocean14c.winds.load_u2).

One reanalysis-era5-single-levels retrieve() call per year, restricted to
2000-2012 to match Miller et al. (2025)'s analysis period exactly (not
extended through the present). For each year: download the raw 6-hourly
file, compute <U^2> = mean(u10^2 + v10^2) at monthly resolution -- the true
second moment the k formula needs, NOT the square of the monthly-mean wind
(CLAUDE.md Sec 4.1: "second moment, not the square of the mean -- this
matters") -- write that small monthly file, then immediately delete the raw
download. This keeps peak disk usage near one year's raw file (~300MB)
rather than the full 13-year archive. Resumable: a year whose processed
output already exists is skipped, so an interrupted run can restart without
re-downloading finished years.

Sampling note: 6-hourly (00/06/12/18Z) is a practical approximation to the
paper's ERA-Interim wind forcing, updated every 90 min (CLAUDE.md Sec 4.5) --
adequate for a monthly second moment, not an exact reproduction of the
original temporal sampling.
"""

import argparse
from pathlib import Path

import xarray as xr

START_YEAR = 2000
END_YEAR = 2012  # inclusive -- Miller et al. (2025)'s analysis period
GRID_RESOLUTION = "1.0/1.0"
HOURS = ["00:00", "06:00", "12:00", "18:00"]

DEFAULT_OUTPUT_DIR = Path("data/processed/era5_u2")


def _time_coord_name(ds: xr.Dataset) -> str:
    """CDS output has used both 'time' and 'valid_time' for the same axis
    across different API/library versions -- accept either."""
    for candidate in ("valid_time", "time"):
        if candidate in ds.coords:
            return candidate
    raise KeyError(f"no recognized time coordinate in dataset (coords: {list(ds.coords)})")


def download_year(year: int, raw_path: Path) -> None:
    """One retrieve() call for a full year of 6-hourly 10m wind components."""
    import cdsapi

    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": HOURS,
            "grid": GRID_RESOLUTION,
            "format": "netcdf",
        },
        str(raw_path),
    )


def compute_monthly_u2(raw_path: Path) -> xr.DataArray:
    """<U^2> per month from one year's raw ERA5 file."""
    with xr.open_dataset(raw_path) as ds:
        time_dim = _time_coord_name(ds)
        u2 = ds["u10"] ** 2 + ds["v10"] ** 2
        monthly = u2.resample({time_dim: "1MS"}).mean()
        monthly = monthly.rename("u2").load()
    return monthly


def process_year(year: int, output_dir=DEFAULT_OUTPUT_DIR, tmp_dir=None) -> Path:
    """Download, process, and delete-raw for one year; skip if already done."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"u2_{year}.nc"

    if output_path.exists():
        return output_path  # resumable

    tmp_dir = Path(tmp_dir) if tmp_dir is not None else output_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw_path = tmp_dir / f"_raw_wind_{year}.nc"

    download_year(year, raw_path)
    try:
        monthly_u2 = compute_monthly_u2(raw_path)
        monthly_u2.to_netcdf(output_path)
    finally:
        raw_path.unlink(missing_ok=True)  # process-then-delete

    return output_path


def download_all(output_dir=DEFAULT_OUTPUT_DIR, start_year=START_YEAR, end_year=END_YEAR):
    for year in range(start_year, end_year + 1):
        print(f"processing {year}...")
        process_year(year, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description="Download and process ERA5 <U^2>")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    args = parser.parse_args()
    download_all(Path(args.output_dir), args.start_year, args.end_year)


if __name__ == "__main__":
    main()
