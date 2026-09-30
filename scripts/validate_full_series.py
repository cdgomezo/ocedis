"""Reconstruct the full 2000-2012 ocean isoflux series and validate it
against Table 1 (SUM row included), Table S2, and the paper's own gridded
product (Ocediseq.nc) -- both temporally (annual series) and spatially
(per-year maps and a difference map).

Usage:
    python3 scripts/validate_full_series.py [--start-year 2000] [--end-year 2012]
                                             [--gamma 0.31] [--map-years 2000,2006,2012]
                                             [--flux-variant reconstruction|table1|both]

--flux-variant chooses which TM5-facing flux product is written and never
changes anything else in this script -- the validation numbers, the CSV and
the figures always describe the reconstruction, because that is the thing
being validated. 'table1' additionally writes the same field rescaled onto
Table 1's published annual values (ocean14c.table1); the two products carry
different `<term>` names and sit side by side in --flux-dir.

Requires: the ERA5 <U^2> data for every requested year already processed
(data/processed/era5_u2/u2_<year>.nc -- see ocean14c.atmosphere_forcing),
plus GLODAP, Takahashi, and Delta_atm raw data (see docs/DATA.md).

Colors follow the Okabe-Ito colorblind-safe qualitative palette throughout,
plus a redundant marker/linestyle encoding per series (never color alone).
Spatial maps use 'viridis' (sequential -- isoflux is one-signed, negative
everywhere in this dataset) and 'RdBu_r' for the ours-minus-theirs
difference map (diverging, zero-centered).
"""

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ocean14c import table1  # noqa: E402
from ocean14c.gas_exchange import DEFAULT_GAMMA  # noqa: E402
from ocean14c.output import write_year  # noqa: E402
from ocean14c.pipeline import cell_area_m2, reconstruct_year  # noqa: E402
from ocean14c.validate import (  # noqa: E402
    TABLE_1_FOA_DELTA_ODIS,
    TABLE_1_SUM,
    load_ocediseq_annual,
    spatial_correlation,
)

# Okabe-Ito colorblind-safe qualitative palette
COLOR_RECON = "#0072B2"  # blue
COLOR_TABLE1 = "#E69F00"  # orange
COLOR_OCEDISEQ = "#009E73"  # bluish green

# Checked in both places: the file lives at the repo root in docs/DATA.md's
# layout, but it has also been moved into data/. Looking only at the root made
# `ocediseq_sum` and `spatial_correlation` vanish from the CSV and the middle
# column of the spatial figure go blank -- silently, since the comparison is
# guarded by `.exists()`.
OCEDISEQ_PATH = next(
    (p for p in (REPO_ROOT / "Ocediseq.nc", REPO_ROOT / "data" / "Ocediseq.nc")
     if p.exists()),
    REPO_ROOT / "Ocediseq.nc",
)


SECONDS_PER_YEAR = 365.25 * 24 * 3600.0
G_PER_MOL_C = 12.011
PG_PER_G = 1e-15


def c14flux_to_pgc_permil_per_yr(c14flux_monthly, lat):
    """Convert the shipped c14flux [umol C_D14C m^-2 s^-1] back to the annual
    per-cell isoflux [PgC permil/yr] that Table 1 reports.

    Exact inverse of ocean14c.isoflux.to_umol_flux composed with
    foa_flux_to_pgc_per_yr:
        x 1e-6      -> mol C_D14C m^-2 s^-1
        x 1000      -> mol C m^-2 s^-1 permil  (undo dimensionless-Delta)
        x s_per_yr  -> mol C m^-2 yr^-1 permil
        x area      -> mol C yr^-1 permil per cell
        x M_C x 1e-15 -> PgC permil / yr per cell
    Then mean over months, matching pipeline.py's annual convention.
    """
    area = cell_area_m2(lat)
    per_month = (
        c14flux_monthly * 1e-6 * 1000.0 * SECONDS_PER_YEAR * area * G_PER_MOL_C * PG_PER_G
    )
    return per_month.mean(dim="month")


def run_reconstruction(years, data_dir, gamma, flux_dir=None,
                       regression_version="original", curve_fit="simplified",
                       flux_variants=("reconstruction",), atm_variant=None):
    rows = []
    annual_fields = {}
    # Kept only for the table1 variant: its scale factors are not known until
    # every year has been reconstructed (out-of-table years take the mean of
    # the in-table ones), so those files cannot be written inside this loop.
    c14flux_by_year = {}
    for year in years:
        era5_path = Path(data_dir) / "processed" / "era5_u2" / f"u2_{year}.nc"
        if not era5_path.exists():
            print(f"  {year}: SKIPPED (no ERA5 data at {era5_path})")
            continue
        print(f"  {year}: reconstructing...")
        result = reconstruct_year(year, data_dir=data_dir, gamma=gamma,
                                  regression_version=regression_version,
                                  curve_fit=curve_fit,
                                  atm_variant=atm_variant,
                                  krige_dir=str(REPO_ROOT / "output"))
        annual_fields[year] = result["annual_isoflux"]

        # Unit cross-check: the shipped umol m-2 s-1 product must reproduce the
        # validated PgC permil/yr number it is derived from. Guards against a
        # stray factor (notably the permil/1000) in to_umol_flux().
        c14flux = result["c14flux_monthly"]
        lat = result["annual_isoflux"]["lat"]
        round_trip = float(
            c14flux_to_pgc_permil_per_yr(c14flux, lat).sum(skipna=True)
        )
        direct = result["global_isoflux"]
        rel = abs(round_trip - direct) / abs(direct)
        if rel > 1e-10:
            raise SystemExit(
                f"ERROR: {year}: c14flux round-trip {round_trip:.4f} != "
                f"global_isoflux {direct:.4f} (rel {rel:.3e}) -- unit "
                f"conversion is inconsistent."
            )
        row_flux_check = rel

        if flux_dir is not None and "reconstruction" in flux_variants:
            path = write_year(year, c14flux, flux_dir,
                              term=table1.TERM_RECONSTRUCTION)
            print(
                f"    wrote {Path(path).name}  "
                f"(mean {float(c14flux.mean(skipna=True)):.4e} umol m-2 s-1, "
                f"unit round-trip ok to {row_flux_check:.1e})"
            )
        if flux_dir is not None and "table1" in flux_variants:
            c14flux_by_year[year] = c14flux

        row = {
            "year": year,
            "global_isoflux": result["global_isoflux"],
            "global_foa": result["global_foa"],
            "global_delta_odis": result["global_delta_odis"],
            "socean_foa": result["socean_foa"],
            "socean_delta_odis": result["socean_delta_odis"],
            "table1_target": TABLE_1_FOA_DELTA_ODIS.get(year),
            "atm_station_nh": result.get("atm_station_nh"),
            "atm_station_sh": result.get("atm_station_sh"),
            "atm_extrapolation_yr": result.get("atm_extrapolation_yr"),
        }

        # Ocediseq.nc only covers 2000-2012; skip the comparison outside that
        # range rather than silently summing an empty selection to zero.
        if OCEDISEQ_PATH.exists() and 2000 <= year <= 2012:
            ocediseq_field = load_ocediseq_annual(OCEDISEQ_PATH, year)
            row["ocediseq_sum"] = float(ocediseq_field.sum(skipna=True))
            ours = result["annual_isoflux"].transpose("lat", "lon")
            theirs = ocediseq_field.transpose("lat", "lon")
            row["spatial_correlation"] = spatial_correlation(ours, theirs)
            annual_fields.setdefault("_ocediseq", {})[year] = ocediseq_field

        rows.append(row)

    df = pd.DataFrame(rows).set_index("year")

    if c14flux_by_year:
        factors = write_table1_variant(df, c14flux_by_year, flux_dir, curve_fit)
        df["table1_scale"] = pd.Series(factors)

    return df, annual_fields


def write_table1_variant(df, c14flux_by_year, flux_dir, curve_fit):
    """Write the Table 1-matched flux files and return {year: scale factor}.

    Separate from the reconstruction pass because the factor for a year Table 1
    does not cover is the mean of the factors for the years it does -- which is
    only known once the whole series has been reconstructed.
    """
    global_isoflux = {y: df.loc[y, "global_isoflux"] for y in c14flux_by_year}
    factors = table1.scale_factors(global_isoflux, curve_fit=curve_fit)

    if table1.uses_reference_fallback(global_isoflux):
        print(
            f"WARNING: no year in this run is covered by Table 1 (2000-2012), so "
            f"every year is scaled by the frozen fallback factor "
            f"{table1.reference_mean_scale(curve_fit):.4f} for curve_fit="
            f"{curve_fit!r}, derived under {table1.REFERENCE_SETTINGS}. If this "
            f"run uses different settings that factor is not the right one -- "
            f"include 2000-2012 in the run to have it recomputed."
        )

    print()
    print("Table 1-matched variant:")
    for year in sorted(c14flux_by_year):
        scaling = table1.describe({year: factors[year]})
        path = write_year(
            year, table1.scale_flux(c14flux_by_year[year], factors[year]),
            flux_dir, term=table1.TERM_TABLE1, scaling=scaling,
        )
        print(f"  wrote {Path(path).name}  x{factors[year]:.4f}")
    return factors


def plot_temporal_comparison(df, out_path):
    fig, (ax_isoflux, ax_corr) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, height_ratios=[2, 1])

    ax_isoflux.plot(
        df.index, df["global_isoflux"], color=COLOR_RECON, marker="o", linestyle="-",
        label="This reconstruction",
    )
    ax_isoflux.plot(
        df.index, df["table1_target"], color=COLOR_TABLE1, marker="s", linestyle="--",
        label="Table 1 (Miller et al. 2025)",
    )
    if "ocediseq_sum" in df.columns:
        ax_isoflux.plot(
            df.index, df["ocediseq_sum"], color=COLOR_OCEDISEQ, marker="^", linestyle=":",
            label="Ocediseq.nc (gridded product)",
        )
    ax_isoflux.set_ylabel("Global FoaΔodis (PgC ‰ / yr)")
    ax_isoflux.set_title("Ocean ¹⁴C disequilibrium isoflux: reconstruction vs. Miller et al. (2025)")
    ax_isoflux.legend(frameon=False)
    ax_isoflux.grid(True, alpha=0.25)
    ax_isoflux.axhline(0, color="0.7", linewidth=0.8)

    if "spatial_correlation" in df.columns:
        ax_corr.plot(df.index, df["spatial_correlation"], color=COLOR_RECON, marker="o")
        ax_corr.set_ylabel("Spatial correlation\nvs. Ocediseq.nc")
        ax_corr.set_ylim(0, 1)
        ax_corr.grid(True, alpha=0.25)
    ax_corr.set_xlabel("Year")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_spatial_maps(annual_fields, map_years, out_path):
    ocediseq_fields = annual_fields.get("_ocediseq", {})
    map_years = [y for y in map_years if y in annual_fields]
    n_rows = len(map_years)
    if n_rows == 0:
        print("no years available for spatial maps; skipping")
        return

    all_vals = np.concatenate(
        [annual_fields[y].values.ravel() for y in map_years]
        + [ocediseq_fields[y].values.ravel() for y in map_years if y in ocediseq_fields]
    )
    vmin, vmax = np.nanpercentile(all_vals, [1, 99])

    all_diffs = np.concatenate(
        [
            (annual_fields[y].transpose("lat", "lon") - ocediseq_fields[y].transpose("lat", "lon")).values.ravel()
            for y in map_years
            if y in ocediseq_fields
        ]
    )
    dmax = float(np.nanpercentile(np.abs(all_diffs), 98))

    # constrained_layout handles cartopy-projected subplots + shared colorbars
    # far more reliably than tight_layout(), which left large uneven gaps
    # between rows when per-axis colorbars were attached inside the loop.
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(13, 3.3 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, year in enumerate(map_years):
        ours = annual_fields[year].transpose("lat", "lon")
        ax = axes[i, 0]
        mesh = ax.pcolormesh(
            ours["lon"], ours["lat"], ours.values, cmap="viridis", vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines(linewidth=0.5)
        ax.set_title(f"{year}: this reconstruction")

        ax2 = axes[i, 1]
        if year in ocediseq_fields:
            theirs = ocediseq_fields[year].transpose("lat", "lon")
            ax2.pcolormesh(
                theirs["lon"], theirs["lat"], theirs.values, cmap="viridis", vmin=vmin, vmax=vmax,
                transform=ccrs.PlateCarree(),
            )
        ax2.coastlines(linewidth=0.5)
        ax2.set_title(f"{year}: Ocediseq.nc")

        ax3 = axes[i, 2]
        diff_mesh = None
        if year in ocediseq_fields:
            theirs = ocediseq_fields[year].transpose("lat", "lon")
            diff = ours - theirs
            diff_mesh = ax3.pcolormesh(
                diff["lon"], diff["lat"], diff.values, cmap="RdBu_r", vmin=-dmax, vmax=dmax,
                transform=ccrs.PlateCarree(),
            )
        ax3.coastlines(linewidth=0.5)
        ax3.set_title(f"{year}: difference")

    fig.colorbar(
        mesh, ax=axes[:, :2], label="Annual isoflux (PgC ‰ / yr / cell)", shrink=0.7, location="bottom",
    )
    if diff_mesh is not None:
        fig.colorbar(diff_mesh, ax=axes[:, 2], label="ours − theirs (PgC ‰ / yr / cell)", shrink=0.7, location="bottom")
    fig.suptitle("Ocean ¹⁴C disequilibrium isoflux: spatial pattern vs. Miller et al. (2025)'s gridded product")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2012)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--map-years", default="2000,2006,2012")
    parser.add_argument(
        "--regression-version", default="original",
        choices=["original", "v2", "glodap2020_npac"],
        help="Surface Delta14C rate-of-change coefficients. 'original' "
             "reproduces Table 1/S2 and is the validated default; "
             "'glodap2020_npac' applies the GLODAPv2.2020 northern North "
             "Pacific correction (docs/SCIENCE.md 8.2).",
    )
    parser.add_argument(
        "--flux-dir",
        default=str(REPO_ROOT / "output"),
        help="Directory for the TM5-facing flux_c14.ocedis.<year>.nc files "
             "(umol C_D14C m-2 s-1). Pass 'none' to skip writing them.",
    )
    parser.add_argument(
        "--flux-variant", default="reconstruction",
        choices=["reconstruction", "table1", "both"],
        help="which TM5-facing flux product(s) to write. 'reconstruction' "
             "(default) writes flux_c14.ocedis.<year>.nc, this project's own "
             "independent reconstruction, unchanged. 'table1' writes "
             "flux_c14.ocedis_table1.<year>.nc, the same field rescaled per "
             "year so its annual global isoflux equals Miller et al. (2025) "
             "Table 1 exactly (amplitude only -- pattern, seasonality and sign "
             "are untouched; see ocean14c.table1). 'both' writes both, side by "
             "side. The two never overwrite each other.",
    )
    parser.add_argument(
        "--atm-geometry", default=None,
        choices=["miller_band", "basu_band", "hemispheric", "hua", "krige"],
        help="which Delta_atm the ocean disequilibrium uses. Omit to keep the "
             "shipped default (miller_band -- NH/SH + 10 permil over 10N-40S), "
             "which every validated number here was built with. 'krige' is the "
             "1x1 kriged station field, the same one TM5's atmos_del now reads, "
             "and is what makes the two consistent; it moves the global ocean "
             "isoflux by about +232 PgC permil/yr. See docs/SCIENCE.md 12.",
    )
    parser.add_argument(
        "--atm-level", default="native", choices=["native", "stations"],
        help="only meaningful with --atm-geometry hua: 'stations' keeps Hua's "
             "shape but takes the absolute level from our own network.",
    )
    parser.add_argument(
        "--curve-fit", default="simplified", choices=["simplified", "ccgcrv"],
        help="atmospheric curve fit. 'simplified' is the validated default; "
             "'ccgcrv' is NOAA's own method (Thoning et al. 1989) and is "
             "required to reach 2024-2025, which extrapolate past the end of "
             "the southern-hemisphere records.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    flux_dir = None if args.flux_dir.lower() == "none" else args.flux_dir
    flux_variants = (
        ("reconstruction", "table1") if args.flux_variant == "both"
        else (args.flux_variant,)
    )

    # None keeps pipeline.reconstruct_year on its original code path, byte for
    # byte -- not merely on an equivalent default. That matters because every
    # Table 1 / Table S2 number in this repo was produced through it.
    atm_variant = (None if args.atm_geometry is None
                   else {"geometry": args.atm_geometry, "level": args.atm_level})

    years = list(range(args.start_year, args.end_year + 1))
    print(f"Reconstructing {years[0]}-{years[-1]} with gamma={args.gamma}, "
          f"regression={args.regression_version}, curve_fit={args.curve_fit}, "
          f"atm={args.atm_geometry or 'default (miller_band)'}...")
    df, annual_fields = run_reconstruction(
        years, args.data_dir, args.gamma, flux_dir=flux_dir,
        regression_version=args.regression_version,
        curve_fit=args.curve_fit,
        flux_variants=flux_variants,
        atm_variant=atm_variant,
    )

    df.to_csv(out_dir / "full_series_results.csv")
    print(f"wrote {out_dir / 'full_series_results.csv'}")

    in_table1 = df["table1_target"].notna()
    computed_sum = df.loc[in_table1, "global_isoflux"].sum()
    print()
    t1_years = df.index[in_table1]
    print(f"Computed SUM ({t1_years.min()}-{t1_years.max()}, Table 1 window): "
          f"{computed_sum:.1f} PgC permil")
    print(f"Table 1 SUM (2000-2012):                     {TABLE_1_SUM} PgC permil")
    print(f"pct error: {100*(computed_sum - TABLE_1_SUM)/TABLE_1_SUM:.1f}%")
    if "spatial_correlation" in df.columns:
        print(f"Mean spatial correlation vs Ocediseq.nc: {df['spatial_correlation'].mean():.3f}")

    plot_temporal_comparison(df, out_dir / "figures" / "temporal_comparison.png")

    map_years = [int(y) for y in args.map_years.split(",")]
    plot_spatial_maps(annual_fields, map_years, out_dir / "figures" / "spatial_comparison.png")


if __name__ == "__main__":
    main()
