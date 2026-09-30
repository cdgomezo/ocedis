"""Comparison and verification figures for the two TM5-facing flux products:
the reconstruction (`flux_c14.ocedis.<year>.nc`) and the Table 1-matched
variant (`flux_c14.ocedis_table1.<year>.nc`, see ocean14c.table1 and
docs/SCIENCE.md 8.3).

Usage:
    python3 scripts/plot_flux_variants.py [--flux-dir output] [--year 2011]
                                          [--out-dir results]

Reads the SHIPPED NetCDF files rather than re-running the pipeline. That is
deliberate: these are verification figures, so they have to describe the bytes
that TM5 will actually be given, not an in-memory array that is one bug away
from what was written. Re-running the pipeline is
`scripts/validate_full_series.py`.

Writes three figures into <out-dir>/figures/ and the underlying table as
<out-dir>/flux_variants_comparison.csv (every figure here has a table view --
the numbers are small enough to read, and the residual panel is at a
precision no eye can check off a plot).

WHAT EACH FIGURE ANSWERS
  1. temporal   -- do the two products differ the way they are supposed to,
                   and does the Table 1 variant sit on Table 1?
  2. spatial    -- is the difference between them really uniform, or did the
                   rescaling deform the field somewhere?
  3. pointwise  -- same question at the level of individual cells and months,
                   where a map cannot resolve a 1% error.

COLOR
Okabe-Ito, as everywhere else in this repo, with redundant marker+linestyle
encoding so identity is never carried by hue alone. Two deviations from
`validate_full_series.py`'s palette, both checked with the six-check
validator (lightness band, chroma floor, adjacent-pair CVD dE, normal-vision
floor, contrast vs surface):
  * the Table 1 series moves from Okabe-Ito orange #E69F00 to #B87500 -- same
    hue, but #E69F00 has contrast 2.25 against white, below the 3.0 floor,
    and it is a reference series that must stay legible;
  * the new Table 1-variant series takes reddish purple #CC79A7, the only
    Okabe-Ito hue that clears CVD dE >= 8 against all three existing series
    (vermillion #D55E00 collapses onto any orange dark enough to pass
    contrast: dE 0.7-3.0 under deuteranopia).
Sequential viridis for the one-signed flux fields and diverging RdBu_r,
zero-centred, for differences.
"""

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ocean14c import table1  # noqa: E402
from ocean14c.validate import (  # noqa: E402
    TABLE_1_FOA_DELTA_ODIS,
    load_ocediseq_annual,
    spatial_correlation,
)
# Reused, not reimplemented: it is the exact inverse of isoflux.to_umol_flux,
# including the permil/1000 that is easy to drop, and validate_full_series.py
# asserts on it every run.
from validate_full_series import c14flux_to_pgc_permil_per_yr  # noqa: E402

COLOR_RECON = "#0072B2"    # blue          -- the reconstruction
COLOR_TABLE1V = "#CC79A7"  # reddish purple -- the Table 1-matched variant
COLOR_TABLE1 = "#B87500"   # orange        -- Table 1's published values
COLOR_OCEDISEQ = "#009E73"  # bluish green  -- the paper's gridded product
INK = "#1a1a19"
MUTED = "#6b6b68"

# Ocediseq.nc moved into data/ at some point; validate_full_series.py still
# looks only at the repo root and silently drops the comparison when it is not
# there. Check both.
OCEDISEQ_CANDIDATES = (REPO_ROOT / "Ocediseq.nc", REPO_ROOT / "data" / "Ocediseq.nc")

FLOAT32_EPS_PPM = np.finfo(np.float32).eps * 1e6  # 0.119 ppm


def find_ocediseq():
    for path in OCEDISEQ_CANDIDATES:
        if path.exists():
            return path
    return None


def load_year(flux_dir, term, year):
    """The shipped field for one year, dims (month, lat, lon)."""
    path = Path(flux_dir) / f"flux_c14.{term}.{year}.nc"
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        return ds["c14flux"].rename({"time": "month"}).load()


def available_years(flux_dir, term):
    return sorted(
        int(p.name.split(".")[-2])
        for p in Path(flux_dir).glob(f"flux_c14.{term}.*.nc")
    )


def build_table(flux_dir):
    """Annual global isoflux [PgC permil/yr] of both products, plus Table 1,
    Ocediseq.nc, the implied scale factor and the residual against Table 1."""
    years = available_years(flux_dir, table1.TERM_RECONSTRUCTION)
    ocediseq = find_ocediseq()
    rows = []
    for year in years:
        recon = load_year(flux_dir, table1.TERM_RECONSTRUCTION, year)
        variant = load_year(flux_dir, table1.TERM_TABLE1, year)
        row = {
            "year": year,
            "recon": float(c14flux_to_pgc_permil_per_yr(recon, recon.lat).sum()),
            "table1_target": TABLE_1_FOA_DELTA_ODIS.get(year),
        }
        if variant is not None:
            row["table1_variant"] = float(
                c14flux_to_pgc_permil_per_yr(variant, variant.lat).sum()
            )
            row["scale_factor"] = row["table1_variant"] / row["recon"]
        if ocediseq is not None and 2000 <= year <= 2012:
            field = load_ocediseq_annual(ocediseq, year)
            row["ocediseq"] = float(field.sum(skipna=True))
            # Scale-invariant by construction, so it is IDENTICAL for the two
            # products -- computed on the variant to make that checkable.
            #
            # Our zeros are masked back to NaN first, Ocediseq's are NOT --
            # that combination is what reproduces the validated numbers in
            # docs/SCIENCE.md 8 (0.9246 for 2000) exactly. Land is 0.0 in the
            # shipped files because TM5 must not be given NaN, and it is 0.0 in
            # Ocediseq.nc too, so leaving ours in adds ~31,500
            # zero-against-zero pairs and inflates r to 0.9383 -- a "better"
            # number that is 49% land. Masking BOTH sides gives 0.9361: it also
            # drops the cells Ocediseq zeroes and we do not, which is a
            # different (defensible, but undocumented) definition.
            annual = c14flux_to_pgc_permil_per_yr(
                variant if variant is not None else recon,
                recon.lat,
            ).transpose("lat", "lon")
            row["spatial_correlation"] = spatial_correlation(
                annual.where(annual != 0), field.transpose("lat", "lon")
            )
        rows.append(row)

    df = pd.DataFrame(rows).set_index("year")
    if "table1_variant" in df:
        # Relative, in ppm: the absolute residual is ~2e-5 PgC permil/yr, which
        # says nothing on its own. Against float32's 0.119 ppm it says the only
        # thing separating the product from Table 1 is how the file stores it.
        df["residual_ppm"] = 1e6 * (df.table1_variant - df.table1_target) / df.table1_target
    return df


def plot_temporal(df, out_path):
    fig, (ax, ax_res, ax_fac) = plt.subplots(
        3, 1, figsize=(9, 9), sharex=True, height_ratios=[3, 1.2, 1.4],
        constrained_layout=True,
    )

    in_window = df.table1_target.notna()
    last_table1_year = int(df.index[in_window].max())

    # Everything right of this line is extrapolated calibration, not Table 1.
    for axis in (ax, ax_fac):
        axis.axvspan(last_table1_year + 0.5, df.index.max() + 0.5,
                     color="0.93", zorder=0)

    ax.axhline(0, color="0.75", linewidth=0.8, zorder=1)
    if "ocediseq" in df:
        ax.plot(df.index, df.ocediseq, color=COLOR_OCEDISEQ, marker="^",
                linestyle=":", markersize=6, linewidth=1.6, zorder=2,
                label="Ocediseq.nc (their gridded product)")
    ax.plot(df.index, df.table1_target, color=COLOR_TABLE1, marker="s",
            linestyle="--", markersize=9, markerfacecolor="none",
            markeredgewidth=1.6, linewidth=1.6, zorder=3,
            label="Table 1, Miller et al. (2025)")
    ax.plot(df.index, df.recon, color=COLOR_RECON, marker="o", linestyle="-",
            markersize=5, linewidth=2, zorder=4,
            label="flux_c14.ocedis (reconstruction)")
    if "table1_variant" in df:
        ax.plot(df.index, df.table1_variant, color=COLOR_TABLE1V, marker="D",
                linestyle="-", markersize=4, linewidth=2, zorder=5,
                label="flux_c14.ocedis_table1 (variant)")

    ax.annotate(
        "no Table 1 beyond 2012:\nthe variant is scaled by the\nmean in-window factor",
        xy=(last_table1_year + 1.2, ax.get_ylim()[1] * 0.62),
        color=MUTED, fontsize=8.5, va="top",
    )
    sign_change = df.index[(df.recon > 0)]
    if len(sign_change):
        first = int(sign_change.min())
        ax.annotate(
            f"sign change ({first}): the ocean\nstops being a ¹⁴C sink",
            xy=(first, float(df.loc[first, "recon"])),
            xytext=(float(df.index.max()) + 0.4, -1000),
            color=INK, fontsize=8.5, ha="right",
            arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8,
                            shrinkA=2, shrinkB=4),
        )

    ax.set_ylabel("Global FoaΔodis  (PgC ‰ / yr)")
    ax.set_title("Annual global ocean ¹⁴C disequilibrium isoflux: "
                 "the two TM5 flux products", fontsize=11)
    # Products first, references after -- plotting order is z-order, which is
    # the reverse of reading order here.
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for l in sorted(
        labels, key=lambda l: (not l.startswith("flux_c14"), l))]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              frameon=False, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.2)

    # --- verification: does the variant actually sit on Table 1? -----------
    ax_res.axhline(0, color="0.75", linewidth=0.8)
    ax_res.axhline(FLOAT32_EPS_PPM, color=MUTED, linestyle="--", linewidth=1)
    ax_res.text(df.index.min(), FLOAT32_EPS_PPM * 1.15,
                f"float32 storage precision ({FLOAT32_EPS_PPM:.2f} ppm)",
                color=MUTED, fontsize=8)
    ax_res.plot(df.index[in_window], df.residual_ppm[in_window],
                color=COLOR_TABLE1V, marker="D", linestyle="none", markersize=5)
    ax_res.set_ylim(-FLOAT32_EPS_PPM * 1.6, FLOAT32_EPS_PPM * 1.9)
    ax_res.set_ylabel("variant − Table 1\n(ppm of target)")
    ax_res.grid(True, alpha=0.2)

    # --- the factor itself --------------------------------------------------
    factors = df.scale_factor
    mean_in = float(factors[in_window].mean())
    ax_fac.axhline(mean_in, color=MUTED, linestyle="--", linewidth=1)
    ax_fac.plot(df.index[in_window], factors[in_window], color=COLOR_TABLE1V,
                marker="D", linestyle="none", markersize=6)
    ax_fac.plot(df.index[~in_window], factors[~in_window], color=COLOR_TABLE1V,
                marker="D", linestyle="none", markersize=6,
                markerfacecolor="none", markeredgewidth=1.4)
    # Direct labels rather than a legend box: filled/open and left/right are
    # already unambiguous, and a box here lands on top of the 2009 point.
    span = float(factors[in_window].max() - factors[in_window].min())
    ax_fac.annotate("fitted per year on Table 1",
                    xy=(2000.4, factors[in_window].min() - 0.35 * span),
                    color=INK, fontsize=8.5)
    ax_fac.annotate("carried forward: mean of the in-window factors",
                    xy=(last_table1_year + 1.0, mean_in - 0.30 * span),
                    color=INK, fontsize=8.5)
    ax_fac.annotate(f"{mean_in:.4f}", xy=(df.index.max() + 0.8, mean_in),
                    color=MUTED, fontsize=8.5, va="bottom")
    ax_fac.set_ylim(factors[in_window].min() - 0.55 * span,
                    factors[in_window].max() + 0.15 * span)
    ax_fac.set_xlim(df.index.min() - 1.2, df.index.max() + 2.6)
    ax_fac.set_ylabel("variant / reconstruction")
    ax_fac.set_xlabel("Year")
    ax_fac.grid(True, alpha=0.2)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_spatial(flux_dir, year, out_path):
    """Four maps. The first two look identical on purpose -- an 11% amplitude
    difference is not visible on a shared colour scale, which is exactly why
    the second row exists: the difference map shows WHERE the rescaling acted
    (everywhere, in proportion to the field) and the ratio map shows that it
    acted by the same number in every cell."""
    recon = load_year(flux_dir, table1.TERM_RECONSTRUCTION, year)
    variant = load_year(flux_dir, table1.TERM_TABLE1, year)
    if recon is None or variant is None:
        print(f"skipping spatial figure: {year} missing from one product")
        return

    a = recon.mean(dim="month")
    b = variant.mean(dim="month")
    ocean = a != 0  # land is written as 0.0, not NaN, for TM5's benefit
    a = a.where(ocean)
    b = b.where(ocean)
    diff = b - a
    ratio = (b / a).where(ocean)

    both = np.concatenate([a.values.ravel(), b.values.ravel()])
    vmin, vmax = np.nanpercentile(both, [1, 99])

    # The difference is one-signed whenever the field is (the factor is a
    # positive scalar), and a diverging map with an unused pole would imply a
    # polarity that is not there. Only fall back to diverging if the year
    # really does straddle zero -- 2024 onward does.
    d_lo, d_hi = np.nanmin(diff.values), np.nanmax(diff.values)
    if d_hi <= 0:
        diff_kw = dict(cmap="Blues_r", vmin=float(np.nanpercentile(diff.values, 1)), vmax=0.0)
    elif d_lo >= 0:
        diff_kw = dict(cmap="Reds", vmin=0.0, vmax=float(np.nanpercentile(diff.values, 99)))
    else:
        dmax = float(np.nanpercentile(np.abs(diff.values), 99))
        diff_kw = dict(cmap="RdBu_r", vmin=-dmax, vmax=dmax)

    factor = float(np.nanmedian(ratio.values))
    ppm = 1e-6 * factor
    dev_ppm = 1e6 * float(np.nanmax(np.abs(ratio.values - factor))) / factor

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 7), subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    panels = (
        (axes[0, 0], a, f"{year}: flux_c14.ocedis (reconstruction)",
         dict(cmap="viridis", vmin=vmin, vmax=vmax), "µmol C_D14C m⁻² s⁻¹"),
        (axes[0, 1], b, f"{year}: flux_c14.ocedis_table1 (variant)",
         dict(cmap="viridis", vmin=vmin, vmax=vmax), "µmol C_D14C m⁻² s⁻¹"),
        (axes[1, 0], diff, f"{year}: variant − reconstruction",
         diff_kw, "µmol C_D14C m⁻² s⁻¹"),
        (axes[1, 1], ratio,
         f"{year}: variant / reconstruction (max spread {dev_ppm:.2f} ppm)",
         dict(cmap="viridis", vmin=factor - ppm, vmax=factor + ppm), "ratio"),
    )
    for axis, field, title, kw, cbar_label in panels:
        mesh = axis.pcolormesh(field.lon, field.lat, field.values,
                               transform=ccrs.PlateCarree(), **kw)
        axis.coastlines(linewidth=0.5)
        axis.set_title(title, fontsize=10)
        cbar = fig.colorbar(mesh, ax=axis, location="bottom", shrink=0.85,
                            pad=0.03, label=cbar_label)
        if cbar_label == "ratio":
            cbar.ax.ticklabel_format(useOffset=False)
            cbar.set_ticks([factor - ppm, factor, factor + ppm])
            cbar.ax.set_xticklabels(["−1 ppm", f"{factor:.4f}", "+1 ppm"])

    fig.suptitle(
        f"{year}: the rescaling is one number, so it deepens the field without "
        f"reshaping it\n"
        f"top row shares a colour scale — the {100*(factor-1):.0f}% difference "
        f"between them is not visible there, which is what the bottom row is for",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_pointwise(flux_dir, year, out_path):
    """The check a map cannot do: every cell and every month scaled by the
    SAME number. A 1% deformation somewhere would be invisible above, and
    fatal for a field whose whole claim is 'amplitude only'."""
    recon = load_year(flux_dir, table1.TERM_RECONSTRUCTION, year)
    variant = load_year(flux_dir, table1.TERM_TABLE1, year)
    if recon is None or variant is None:
        print(f"skipping pointwise figure: {year} missing from one product")
        return

    x = recon.values.ravel()
    y = variant.values.ravel()
    ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-12)
    x, y = x[ok], y[ok]
    ratio = y / x
    factor = float(np.median(ratio))
    slope = float(np.sum(x * y) / np.sum(x * x))  # through the origin

    fig, (ax_cycle, ax_scatter, ax_hist) = plt.subplots(
        1, 3, figsize=(14, 4.2), constrained_layout=True
    )

    months = np.arange(1, 13)
    ax_cycle.plot(months, recon.mean(dim=("lat", "lon")), color=COLOR_RECON,
                  marker="o", linestyle="-", markersize=6, linewidth=2,
                  label="flux_c14.ocedis")
    ax_cycle.plot(months, variant.mean(dim=("lat", "lon")), color=COLOR_TABLE1V,
                  marker="D", linestyle="-", markersize=5, linewidth=2,
                  label="flux_c14.ocedis_table1")
    ax_cycle.set_xlabel("Month")
    ax_cycle.set_ylabel("Global mean flux  (µmol m⁻² s⁻¹)")
    ax_cycle.set_title(f"{year}: seasonal cycle", fontsize=10)
    ax_cycle.legend(frameon=False, fontsize=9)
    ax_cycle.grid(True, alpha=0.2)

    span = np.array([x.min(), x.max()])
    ax_scatter.plot(span, factor * span, color=INK, linewidth=2.5,
                    linestyle="-", zorder=1)
    step = max(1, x.size // 25000)
    ax_scatter.plot(x[::step], y[::step], linestyle="none", marker=".",
                    markersize=3, alpha=0.35, color=COLOR_RECON, zorder=2)
    ax_scatter.annotate(
        f"ink line: y = {factor:.4f} x\n"
        f"points: {x.size:,} ocean cells × months",
        xy=(0.03, 0.93), xycoords="axes fraction", color=INK, fontsize=8.5,
        va="top")
    ax_scatter.set_xlabel("flux_c14.ocedis  (µmol m⁻² s⁻¹)")
    ax_scatter.set_ylabel("flux_c14.ocedis_table1")
    ax_scatter.set_title(
        f"{year}: every cell × every month\n"
        f"slope through origin {slope:.6f}", fontsize=10)
    ax_scatter.grid(True, alpha=0.2)

    dev_ppm = 1e6 * (ratio - factor) / factor
    ax_hist.hist(dev_ppm, bins=80, color=COLOR_TABLE1V)
    ax_hist.axvline(0, color=INK, linewidth=1)
    ax_hist.set_xlabel(f"ratio deviation from {factor:.4f}  (ppm)")
    ax_hist.set_ylabel("cells × months")
    ax_hist.set_yscale("log")
    ax_hist.annotate(
        "three spikes, one float32 ulp apart:\n"
        "the only thing that varies across\n"
        "the field is how the product rounds",
        xy=(0.97, 0.93), xycoords="axes fraction", color=INK, fontsize=8.5,
        va="top", ha="right")
    ax_hist.set_title(
        f"{year}: spread of the pointwise ratio\n"
        f"max |dev| {np.abs(dev_ppm).max():.1f} ppm  "
        f"(float32 ≈ {FLOAT32_EPS_PPM:.2f} ppm/op)", fontsize=10)
    ax_hist.grid(True, alpha=0.2)

    fig.suptitle(
        "Amplitude only: one scalar multiplies the whole year, so the pattern, "
        "the seasonal cycle and the sign of every cell are the reconstruction's",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux-dir", default=str(REPO_ROOT / "output"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--year", type=int, default=2011,
                        help="year for the spatial and pointwise figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    df = build_table(args.flux_dir)
    if "table1_variant" not in df:
        raise SystemExit(
            f"no flux_c14.{table1.TERM_TABLE1}.<year>.nc in {args.flux_dir} -- "
            f"produce them first with "
            f"`scripts/validate_full_series.py --flux-variant both`"
        )

    csv_path = out_dir / "flux_variants_comparison.csv"
    df.to_csv(csv_path)
    print(f"wrote {csv_path}")
    print()
    print(df.round(4).to_string())
    print()
    in_window = df.table1_target.notna()
    print(f"max |residual| against Table 1: {df.residual_ppm.abs().max():.4f} ppm "
          f"(float32 storage precision {FLOAT32_EPS_PPM:.3f} ppm)")
    print(f"scale factor: {df.scale_factor[in_window].min():.4f}-"
          f"{df.scale_factor[in_window].max():.4f} in window, "
          f"{df.scale_factor[~in_window].mean():.4f} carried forward")
    if "spatial_correlation" in df:
        print(f"spatial correlation vs Ocediseq.nc: "
              f"{df.spatial_correlation.mean():.3f} mean "
              f"(identical for both products -- it is scale-invariant)")
    print()

    plot_temporal(df, out_dir / "figures" / "flux_variants_temporal.png")
    plot_spatial(args.flux_dir, args.year,
                 out_dir / "figures" / "flux_variants_spatial.png")
    plot_pointwise(args.flux_dir, args.year,
                   out_dir / "figures" / "flux_variants_pointwise.png")


if __name__ == "__main__":
    main()
