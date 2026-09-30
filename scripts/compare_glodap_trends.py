"""Independent validation of the surface Delta14C rate-of-change regression
against GLODAPv2.2020 repeat-section observations.

Why this exists
---------------
`ocean14c.regression.rate_of_change()` is the single most load-bearing
empirical step in this pipeline: it projects the 1995 GLODAP surface field
forward in time (docs/SCIENCE.md Sec 3.2-3.3), and everything downstream
inherits its error. Until now it was validated only against Miller et al.
(2025)'s own products (Table 1/S2, Ocediseq.nc) -- i.e. against the same
paper the regression came from.

This script tests it against genuinely independent data: discrete bottle
Delta14C from GLODAPv2.2020 (Olsen et al. 2020) along the A16 (Atlantic
~30W) and P16 (Pacific ~150W) repeat sections, following the approach of
Pookkandy, Graven & Martin (2023, Clim Dyn 61:3223-3235), who binned by
2 degrees latitude and compared decadal means.

Three estimates of the surface Delta14C trend [permil/decade] are compared:

  observed  GLODAPv2.2020 bottles, least-squares trend per latitude bin
  ocedis    rate_of_change() evaluated on the 1995 GLODAP field
  nemo      Pookkandy et al. (2023) NEMO "Hist" run, same trend fit

Caveats, all real:
  * GLODAP c14 is NOT secondary-quality-controlled (it carries G2c14f and
    G2c14err but no G2c14qc), so inter-cruise offsets are uncorrected.
  * A16 has no North Atlantic coverage after the 1990s -- a genuine
    observational gap, noted by Pookkandy et al. too. Bins there drop out.
  * The NEMO run is biased low by >20 permil in the interior and
    underestimates decadal change by ~5 permil/decade (their Sec 3.6), so
    it is context, not a truth standard. GLODAP is the anchor.

Usage
-----
    python3 scripts/compare_glodap_trends.py
    python3 scripts/compare_glodap_trends.py --depth-max 200
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ocean14c.glodap import surface_delta_1995  # noqa: E402
from ocean14c.regions import northern_north_pacific_mask, southern_ocean_mask  # noqa: E402
from ocean14c.regression import rate_of_change  # noqa: E402

GLODAP_V2 = (
    REPO_ROOT
    / "data/raw/GLODAPv2.2020/0210813/2.3/data/0-data/GLODAPv2.2020_Merged_Master_File.csv"
)
NEMO_HIST = REPO_ROOT / "Pookkandy2023_data/Hist.D14C.yearly.NEMO-C14a.1850-2020.nc"
NEMO_NYF = REPO_ROOT / "Pookkandy2023_data/Hist_NYF.D14C.yearly.NEMO-C14a.1850-2020.nc"

# Okabe-Ito colorblind-safe palette, consistent with validate_full_series.py
C_OBS = "#000000"
C_OCEDIS = "#0072B2"
C_NEMO = "#D55E00"

SECTIONS = {
    "P16 (Pacific ~150W)": -150.0,
    "A16 (Atlantic ~30W)": -30.0,
}

YEAR_MIN, YEAR_MAX = 1990, 2018  # GLODAP c14 coverage ends 2018
LAT_BIN = 2.0                    # Pookkandy et al. binning
MIN_SAMPLES = 6                  # per bin, to fit a trend
MIN_SPAN_YEARS = 12              # per bin, to fit a trend


def wrap180(lon):
    return ((np.asarray(lon) + 180) % 360) - 180


def load_glodap_v2(path, depth_max, lon_tol):
    """Bottle Delta14C along the two sections, flagged good, 1990-2018."""
    cols = [
        "G2cruise", "G2year", "G2latitude", "G2longitude",
        "G2depth", "G2c14", "G2c14f",
    ]
    df = pd.read_csv(path, usecols=cols, na_values=[-9999, -999])
    df = df[df.G2c14.notna() & (df.G2c14f == 2)]
    df = df[(df.G2year >= YEAR_MIN) & (df.G2year <= YEAR_MAX)]
    df = df[df.G2depth <= depth_max]
    df["lon180"] = wrap180(df.G2longitude)

    keep = []
    for name, lonc in SECTIONS.items():
        sel = df[np.abs(df.lon180 - lonc) <= lon_tol].copy()
        sel["section"] = name
        keep.append(sel)
    return pd.concat(keep, ignore_index=True)


def trend_per_bin(df):
    """Least-squares Delta14C trend [permil/decade] per 2-degree latitude bin."""
    edges = np.arange(-90, 90 + LAT_BIN, LAT_BIN)
    df = df.copy()
    df["bin"] = pd.cut(df.G2latitude, edges, labels=False)

    rows = []
    for (section, b), g in df.groupby(["section", "bin"], observed=True):
        if len(g) < MIN_SAMPLES:
            continue
        span = g.G2year.max() - g.G2year.min()
        if span < MIN_SPAN_YEARS:
            continue
        slope, intercept = np.polyfit(g.G2year, g.G2c14, 1)
        # 1-sigma on the slope, from the residual scatter
        resid = g.G2c14 - (slope * g.G2year + intercept)
        dof = max(len(g) - 2, 1)
        ss_x = ((g.G2year - g.G2year.mean()) ** 2).sum()
        se = np.sqrt((resid**2).sum() / dof / ss_x) if ss_x > 0 else np.nan
        rows.append(
            {
                "section": section,
                "lat": edges[int(b)] + LAT_BIN / 2,
                "obs_rate": slope * 10.0,
                "obs_err": se * 10.0,
                "n": len(g),
                "span": span,
                "n_cruises": g.G2cruise.nunique(),
            }
        )
    return pd.DataFrame(rows).sort_values(["section", "lat"]).reset_index(drop=True)


def ocedis_rate_field():
    """rate_of_change() on the 1995 GLODAP field -> DataArray [permil/yr]."""
    d95 = surface_delta_1995(REPO_ROOT / "data/raw/GLODAP")
    is_cold = southern_ocean_mask(d95) | northern_north_pacific_mask(d95)
    return xr.DataArray(
        rate_of_change(d95, is_cold), coords=d95.coords, dims=d95.dims
    )


def nemo_surface_trend(path, lat_target=None):
    """Surface Delta14C trend [permil/decade] from a NEMO run, 1990-2018.

    Reads only the surface level for the needed years -- the files are 3.3 GB
    each and the full array would be pointless to load.
    """
    ds = xr.open_dataset(path, decode_times=True)
    years = ds["time_counter"].dt.year
    sel = ds["D14C"].isel(deptht=0).sel(
        time_counter=(years >= YEAR_MIN) & (years <= YEAR_MAX)
    )
    yr = sel["time_counter"].dt.year.values.astype(float)
    data = sel.values  # (time, lat, lon)

    # Vectorised least-squares slope over the time axis
    y = data.reshape(len(yr), -1)
    valid = np.isfinite(y).all(axis=0)
    slope = np.full(y.shape[1], np.nan)
    if valid.any():
        x = yr - yr.mean()
        yy = y[:, valid]
        slope[valid] = (x[:, None] * (yy - yy.mean(axis=0))).sum(axis=0) / (x**2).sum()
    slope = slope.reshape(data.shape[1], data.shape[2]) * 10.0

    return xr.DataArray(
        slope,
        coords={"lat": ds["lat"].values, "lon": wrap180(ds["lon"].values)},
        dims=("lat", "lon"),
    ).sortby("lon")


def nemo_circulation_variability(hist_path, nyf_path):
    """std over 1990-2018 of surface (Hist - Hist_NYF) Delta14C [permil].

    Hist is forced with inter-annually varying reanalysis; Hist_NYF repeats a
    neutral year. Their difference isolates circulation-driven variability --
    Pookkandy et al. Sec 3.5 / Fig 10.
    """
    out = []
    for path in (hist_path, nyf_path):
        ds = xr.open_dataset(path, decode_times=True)
        years = ds["time_counter"].dt.year
        sel = ds["D14C"].isel(deptht=0).sel(
            time_counter=(years >= YEAR_MIN) & (years <= YEAR_MAX)
        )
        out.append(sel.values)
    diff = out[0] - out[1]
    ds = xr.open_dataset(hist_path)
    return xr.DataArray(
        np.nanstd(diff, axis=0),
        coords={"lat": ds["lat"].values, "lon": wrap180(ds["lon"].values)},
        dims=("lat", "lon"),
    ).sortby("lon")


def sample_along_section(field, lat_values, lonc, lon_tol):
    """Zonal mean of `field` in a lon window, at each requested latitude."""
    sub = field.sel(lon=slice(lonc - lon_tol, lonc + lon_tol))
    out = []
    for lat in lat_values:
        band = sub.sel(lat=slice(lat - LAT_BIN / 2, lat + LAT_BIN / 2))
        vals = band.values[np.isfinite(band.values)]
        out.append(float(vals.mean()) if vals.size else np.nan)
    return np.array(out)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_rate_vs_latitude(df, out_path, depth_max):
    sections = [s for s in SECTIONS if (df.section == s).any()]
    fig, axes = plt.subplots(
        len(sections), 1, figsize=(9, 4.2 * len(sections)), sharex=True
    )
    if len(sections) == 1:
        axes = [axes]

    for ax, section in zip(axes, sections):
        d = df[df.section == section].sort_values("lat")
        ax.axhline(0, color="0.75", lw=0.8)
        ax.errorbar(
            d.lat, d.obs_rate, yerr=d.obs_err, color=C_OBS, marker="o", ms=4,
            lw=1.2, capsize=2, linestyle="-", label="GLODAPv2.2020 observed",
        )
        ax.plot(
            d.lat, d.ocedis_rate, color=C_OCEDIS, marker="s", ms=4, lw=1.6,
            linestyle="--", label="Ocedis regression (this repo)",
        )
        ax.plot(
            d.lat, d.nemo_rate, color=C_NEMO, marker="^", ms=4, lw=1.2,
            linestyle=":", label="NEMO Hist (Pookkandy 2023)",
        )
        ax.set_ylabel("Surface Δ¹⁴C trend\n(‰ per decade)")
        ax.set_title(f"{section}   —   {len(d)} latitude bins with sufficient data")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

    axes[-1].set_xlabel("Latitude (°N)")
    fig.suptitle(
        f"Surface Δ¹⁴C rate of change, 1990–2018 (bottles ≤ {depth_max} m)\n"
        "Independent test of ocean14c.regression.rate_of_change()",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_scatter(df, out_path, depth_max):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    lim = [
        np.nanmin([df.obs_rate.min(), df.ocedis_rate.min(), df.nemo_rate.min()]) - 3,
        np.nanmax([df.obs_rate.max(), df.ocedis_rate.max(), df.nemo_rate.max()]) + 3,
    ]

    for ax, (col, label, color) in zip(
        axes,
        [("ocedis_rate", "Ocedis regression", C_OCEDIS),
         ("nemo_rate", "NEMO Hist (Pookkandy 2023)", C_NEMO)],
    ):
        d = df.dropna(subset=["obs_rate", col])
        sc = ax.scatter(
            d.obs_rate, d[col], c=d.lat, cmap="coolwarm", s=34,
            edgecolor="k", linewidth=0.3, vmin=-70, vmax=70,
        )
        ax.plot(lim, lim, "k--", lw=1, label="1:1")
        if len(d) > 2:
            r = np.corrcoef(d.obs_rate, d[col])[0, 1]
            bias = (d[col] - d.obs_rate).mean()
            ax.set_title(f"{label}\nr = {r:.2f},  bias = {bias:+.1f} ‰/dec,  n = {len(d)}")
        ax.set_xlabel("GLODAP observed (‰/decade)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="upper left", fontsize=9)

    axes[0].set_ylabel("Estimated (‰/decade)")
    fig.colorbar(sc, ax=axes, label="Latitude (°N)", shrink=0.85)
    fig.suptitle(
        f"Predicted vs observed surface Δ¹⁴C trend (bottles ≤ {depth_max} m)"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_circulation_test(df, out_path, depth_max):
    """Does the Ocedis-minus-observed discrepancy concentrate where
    circulation-driven variability is largest?"""
    d = df.dropna(subset=["obs_rate", "ocedis_rate", "circ_var"]).copy()
    d["resid"] = np.abs(d.ocedis_rate - d.obs_rate)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax1.axhline(0, color="0.75", lw=0.8)
    for section, marker in zip(SECTIONS, ["o", "s"]):
        s = d[d.section == section]
        if not len(s):
            continue
        ax1.plot(
            s.lat, s.ocedis_rate - s.obs_rate, marker=marker, ms=5, lw=1.2,
            label=section,
        )
    ax1b = ax1.twinx()
    for section, ls in zip(SECTIONS, ["-", "--"]):
        s = d[d.section == section]
        if not len(s):
            continue
        ax1b.plot(s.lat, s.circ_var, color="0.55", ls=ls, lw=1.4)
    ax1b.set_ylabel("Circulation-driven σ(Δ¹⁴C) (‰)\nNEMO Hist − Hist_NYF (grey)", color="0.4")
    ax1.set_xlabel("Latitude (°N)")
    ax1.set_ylabel("Ocedis − observed (‰/decade)")
    ax1.set_title("Discrepancy and circulation variability vs. latitude")
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False, fontsize=9, loc="lower left")

    sc = ax2.scatter(
        d.circ_var, d.resid, c=d.lat, cmap="coolwarm", s=38,
        edgecolor="k", linewidth=0.3, vmin=-70, vmax=70,
    )
    if len(d) > 2:
        r = np.corrcoef(d.circ_var, d.resid)[0, 1]
        m, b = np.polyfit(d.circ_var, d.resid, 1)
        xs = np.linspace(d.circ_var.min(), d.circ_var.max(), 10)
        ax2.plot(xs, m * xs + b, "k--", lw=1)
        ax2.set_title(f"|Ocedis − observed| vs circulation variability\nr = {r:.2f}, n = {len(d)}")
    ax2.set_xlabel("Circulation-driven σ(Δ¹⁴C) (‰)")
    ax2.set_ylabel("|Ocedis − observed| (‰/decade)")
    ax2.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax2, label="Latitude (°N)")

    fig.suptitle(
        "Does the regression fail where circulation variability is largest?  "
        f"(bottles ≤ {depth_max} m)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth-max", type=float, default=30.0,
                   help="max bottle depth [m]; 30 matches surface_delta_1995(), "
                        "200 matches Pookkandy's 'shallow' layer")
    p.add_argument("--lon-tol", type=float, default=5.0)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    tag = f"{int(args.depth_max)}m"

    print(f"Loading GLODAPv2.2020 (depth <= {args.depth_max} m)...")
    bottles = load_glodap_v2(GLODAP_V2, args.depth_max, args.lon_tol)
    print(f"  {len(bottles)} bottles, {bottles.G2cruise.nunique()} cruises")

    df = trend_per_bin(bottles)
    print(f"  {len(df)} latitude bins with >= {MIN_SAMPLES} samples "
          f"spanning >= {MIN_SPAN_YEARS} yr")

    print("Evaluating Ocedis regression...")
    ocedis = ocedis_rate_field().transpose("lat", "lon")

    print("Fitting NEMO Hist trend (surface, 1990-2018)...")
    nemo = nemo_surface_trend(NEMO_HIST)

    print("Computing NEMO circulation-driven variability (Hist - Hist_NYF)...")
    circ = nemo_circulation_variability(NEMO_HIST, NEMO_NYF)

    for section, lonc in SECTIONS.items():
        m = df.section == section
        if not m.any():
            continue
        lats = df.loc[m, "lat"].values
        df.loc[m, "ocedis_rate"] = sample_along_section(ocedis, lats, lonc, args.lon_tol) * 10.0
        df.loc[m, "nemo_rate"] = sample_along_section(nemo, lats, lonc, args.lon_tol)
        df.loc[m, "circ_var"] = sample_along_section(circ, lats, lonc, args.lon_tol)

    csv = out_dir / f"glodap_trend_comparison_{tag}.csv"
    df.to_csv(csv, index=False)
    print(f"wrote {csv}")

    print("\n" + "=" * 72)
    print(f"SURFACE Δ14C TREND, 1990-2018  (bottles <= {args.depth_max:.0f} m)")
    print("=" * 72)
    for section in SECTIONS:
        d = df[(df.section == section)].dropna(subset=["obs_rate", "ocedis_rate"])
        if not len(d):
            print(f"\n{section}: no bins with sufficient data")
            continue
        r = np.corrcoef(d.obs_rate, d.ocedis_rate)[0, 1] if len(d) > 2 else np.nan
        print(f"\n{section}   ({len(d)} bins, {d.lat.min():+.0f}..{d.lat.max():+.0f} lat)")
        print(f"   observed  median {np.median(d.obs_rate):+7.1f} permil/decade")
        print(f"   ocedis    median {np.median(d.ocedis_rate):+7.1f}   "
              f"bias {np.mean(d.ocedis_rate - d.obs_rate):+.1f}   r {r:+.2f}")
        dn = d.dropna(subset=["nemo_rate"])
        if len(dn) > 2:
            rn = np.corrcoef(dn.obs_rate, dn.nemo_rate)[0, 1]
            print(f"   nemo      median {np.median(dn.nemo_rate):+7.1f}   "
                  f"bias {np.mean(dn.nemo_rate - dn.obs_rate):+.1f}   r {rn:+.2f}")

    dd = df.dropna(subset=["obs_rate", "ocedis_rate", "circ_var"]).copy()
    if len(dd) > 2:
        dd["resid"] = np.abs(dd.ocedis_rate - dd.obs_rate)
        rc = np.corrcoef(dd.circ_var, dd.resid)[0, 1]
        print(f"\ncirculation-variability test: corr(|ocedis-obs|, sigma_circ) "
              f"= {rc:+.2f}  (n={len(dd)})")
    print("=" * 72 + "\n")

    fig_rate_vs_latitude(df, out_dir / "figures" / f"glodap_trend_vs_latitude_{tag}.png", args.depth_max)
    fig_scatter(df, out_dir / "figures" / f"glodap_trend_scatter_{tag}.png", args.depth_max)
    fig_circulation_test(df, out_dir / "figures" / f"glodap_circulation_test_{tag}.png", args.depth_max)


if __name__ == "__main__":
    main()
