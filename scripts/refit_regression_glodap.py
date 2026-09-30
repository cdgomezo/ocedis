"""Refit the surface Delta14C rate-of-change regression from GLODAPv2.2020,
per population, over the whole ocean rather than a single meridional section.

Motivation
----------
`scripts/compare_glodap_trends.py` found that along P16 the published cold
coefficients over-predict the northern North Pacific decline by a factor of
~2 (bias -21.3 permil/decade, RMSE 22.0), with Pookkandy et al. (2023)'s
NEMO run independently agreeing with the observations. That test used one
line at 150W. This script widens it to every GLODAP bottle in each
population, so a coefficient revision rests on the whole region.

Method
------
1. Take GLODAPv2.2020 bottles with valid, flagged-good Delta14C, 1990-2018,
   above `--depth-max`.
2. Bin them into `--box-deg` lat/lon boxes. In each box with enough samples
   spanning enough years, least-squares fit Delta14C vs year -> observed
   rate [permil/yr]. Boxes, not raw bottles, are the regression unit, so
   heavily-sampled sites don't dominate.
3. Look up Delta14C_1995 (GLODAPv1 gridded, the pipeline's own starting
   field) at each box centre, and assign the box to a population using
   ocean14c.regions.
4. Regress observed rate vs Delta14C_1995 per population and compare with
   the published coefficients.

This deliberately mirrors what Lindsay (2016) / Miller et al. (2025) SI
Fig. S2 did, with newer and independently-sourced observations.

Usage
-----
    python3 scripts/refit_regression_glodap.py
    python3 scripts/refit_regression_glodap.py --depth-max 200 --box-deg 4
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ocean14c.glodap import surface_delta_1995  # noqa: E402
from ocean14c.regions import (  # noqa: E402
    northern_north_pacific_mask,
    southern_ocean_mask,
)
from ocean14c.regression import COEFFICIENTS  # noqa: E402

GLODAP_V2 = (
    REPO_ROOT
    / "data/raw/GLODAPv2.2020/0210813/2.3/data/0-data/GLODAPv2.2020_Merged_Master_File.csv"
)

YEAR_MIN, YEAR_MAX = 1990, 2018
C_NPAC = "#D55E00"
C_SO = "#0072B2"
C_WARM = "#E69F00"


def wrap180(lon):
    return ((np.asarray(lon) + 180) % 360) - 180


def load_bottles(depth_max):
    cols = ["G2cruise", "G2year", "G2latitude", "G2longitude", "G2depth", "G2c14", "G2c14f"]
    df = pd.read_csv(GLODAP_V2, usecols=cols, na_values=[-9999, -999])
    df = df[df.G2c14.notna() & (df.G2c14f == 2)]
    df = df[(df.G2year >= YEAR_MIN) & (df.G2year <= YEAR_MAX) & (df.G2depth <= depth_max)]
    df["lon180"] = wrap180(df.G2longitude)
    return df


def box_trends(df, box_deg, min_samples, min_span, min_cruises):
    """Least-squares Delta14C trend per lat/lon box."""
    d = df.copy()
    d["blat"] = np.floor(d.G2latitude / box_deg) * box_deg + box_deg / 2
    d["blon"] = np.floor(d.lon180 / box_deg) * box_deg + box_deg / 2

    rows = []
    for (blat, blon), g in d.groupby(["blat", "blon"]):
        if len(g) < min_samples:
            continue
        if g.G2year.max() - g.G2year.min() < min_span:
            continue
        if g.G2cruise.nunique() < min_cruises:
            continue
        slope, intercept = np.polyfit(g.G2year, g.G2c14, 1)
        resid = g.G2c14 - (slope * g.G2year + intercept)
        ss_x = ((g.G2year - g.G2year.mean()) ** 2).sum()
        se = np.sqrt((resid**2).sum() / max(len(g) - 2, 1) / ss_x) if ss_x > 0 else np.nan
        rows.append(
            {
                "lat": blat, "lon": blon,
                "rate": slope,           # permil/yr
                "rate_err": se,
                "n": len(g),
                "span": g.G2year.max() - g.G2year.min(),
                "n_cruises": g.G2cruise.nunique(),
            }
        )
    return pd.DataFrame(rows)


def attach_population(boxes):
    """delta_1995 at each box centre + population label, using the pipeline's
    own field and region masks."""
    d95 = surface_delta_1995(REPO_ROOT / "data/raw/GLODAP").transpose("lat", "lon")
    so = southern_ocean_mask(d95).transpose("lat", "lon")
    npac = northern_north_pacific_mask(d95).transpose("lat", "lon")

    out = boxes.copy()
    vals, pops = [], []
    for _, r in out.iterrows():
        sel = dict(lat=r.lat, lon=r.lon)
        try:
            v = float(d95.sel(**sel, method="nearest"))
            in_so = bool(so.sel(**sel, method="nearest"))
            in_np = bool(npac.sel(**sel, method="nearest"))
        except Exception:
            v, in_so, in_np = np.nan, False, False
        vals.append(v)
        pops.append("npac" if in_np else ("socean" if in_so else "warm"))
    out["d95"] = vals
    out["pop"] = pops
    return out.dropna(subset=["d95"])


def weighted_fit(x, y, w=None):
    """Least-squares slope/intercept with optional weights, plus r and n."""
    if len(x) < 3:
        return np.nan, np.nan, np.nan, len(x)
    if w is None:
        m, b = np.polyfit(x, y, 1)
    else:
        m, b = np.polyfit(x, y, 1, w=w)
    r = np.corrcoef(x, y)[0, 1]
    return m, b, r, len(x)


def fig_regression(df, out_path, fits, depth_max, box_deg):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    order = [
        ("npac", "Northern North Pacific", C_NPAC, "cold"),
        ("socean", "Southern Ocean", C_SO, "cold"),
        ("warm", "Warm (everywhere else)", C_WARM, "warm"),
    ]
    for ax, (pop, label, color, published_key) in zip(axes, order):
        s = df[df["pop"] == pop]
        ax.axhline(0, color="0.8", lw=0.8)
        ax.scatter(s.d95, s.rate, s=18, color=color, edgecolor="k",
                   linewidth=0.3, alpha=0.75, label=f"GLODAPv2.2020 boxes (n={len(s)})")

        xs = np.linspace(df.d95.min(), df.d95.max(), 50)
        pm, pb = COEFFICIENTS["original"][published_key]
        ax.plot(xs, pm * xs + pb, "k--", lw=1.6,
                label=f"published {published_key}: {pm:+.4f}·Δ{pb:+.4f}")

        m, b, r, n = fits[pop]
        if np.isfinite(m):
            ax.plot(xs, m * xs + b, color=color, lw=2.0,
                    label=f"GLODAP refit: {m:+.4f}·Δ{b:+.4f}  (r={r:+.2f})")

        ax.set_title(label)
        ax.set_xlabel("Δ¹⁴C₁₉₉₅ (‰)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    axes[0].set_ylabel("Observed dΔ¹⁴C/dt (‰ / yr)")
    fig.suptitle(
        f"Surface Δ¹⁴C rate of change vs Δ¹⁴C₁₉₉₅, refit from GLODAPv2.2020 1990–2018\n"
        f"(bottles ≤ {depth_max:.0f} m, {box_deg:.0f}° boxes) — published lines are "
        f"Lindsay (2016) / Miller et al. (2025) SI Fig. S2"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_map(df, out_path, depth_max):
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"npac": C_NPAC, "socean": C_SO, "warm": C_WARM}
    for pop, color in colors.items():
        s = df[df["pop"] == pop]
        ax.scatter(s.lon, s.lat, c=color, s=22, edgecolor="k", linewidth=0.3,
                   label=f"{pop} (n={len(s)})")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"GLODAPv2.2020 boxes with a fittable trend (≤ {depth_max:.0f} m), by population")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth-max", type=float, default=200.0)
    p.add_argument("--box-deg", type=float, default=4.0)
    p.add_argument("--min-samples", type=int, default=6)
    p.add_argument("--min-span", type=int, default=12)
    p.add_argument("--min-cruises", type=int, default=2)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    print(f"Loading GLODAPv2.2020 bottles (<= {args.depth_max:.0f} m, {YEAR_MIN}-{YEAR_MAX})...")
    bottles = load_bottles(args.depth_max)
    print(f"  {len(bottles)} bottles, {bottles.G2cruise.nunique()} cruises")

    boxes = box_trends(bottles, args.box_deg, args.min_samples, args.min_span, args.min_cruises)
    print(f"  {len(boxes)} boxes with >= {args.min_samples} samples, "
          f">= {args.min_span} yr span, >= {args.min_cruises} cruises")

    df = attach_population(boxes)
    print(f"  {len(df)} boxes with a Delta14C_1995 value")
    print(df["pop"].value_counts().to_string())

    fits = {}
    for pop in ("npac", "socean", "warm"):
        s = df[df["pop"] == pop]
        fits[pop] = weighted_fit(s.d95.values, s.rate.values)

    csv = out_dir / f"glodap_regression_refit_{int(args.depth_max)}m.csv"
    df.to_csv(csv, index=False)
    print(f"wrote {csv}")

    print("\n" + "=" * 78)
    print("REFIT: rate [permil/yr] = m * Delta14C_1995 + b")
    print("=" * 78)
    print(f"{'population':<22} {'m':>10} {'b':>10} {'r':>7} {'n':>5}   published (m, b)")
    for pop, published_key in (("npac", "cold"), ("socean", "cold"), ("warm", "warm")):
        m, b, r, n = fits[pop]
        pm, pb = COEFFICIENTS["original"][published_key]
        print(f"{pop:<22} {m:>10.4f} {b:>10.4f} {r:>7.2f} {n:>5}   "
              f"{published_key}: ({pm:+.4f}, {pb:+.4f})")

    # What each choice implies for the northern North Pacific specifically
    s = df[df["pop"] == "npac"]
    if len(s) >= 3:
        print("\nNorthern North Pacific, predicted vs observed rate [permil/decade]:")
        for label, (m, b) in (
            ("published cold", COEFFICIENTS["original"]["cold"]),
            ("published warm", COEFFICIENTS["original"]["warm"]),
            ("GLODAP refit  ", fits["npac"][:2]),
        ):
            pred = (m * s.d95 + b) * 10
            obs = s.rate * 10
            print(f"  {label}: bias {np.mean(pred - obs):+7.1f}   "
                  f"RMSE {np.sqrt(np.mean((pred - obs) ** 2)):6.1f}")
    print("=" * 78 + "\n")

    tag = f"{int(args.depth_max)}m"
    fig_regression(df, out_dir / "figures" / f"glodap_regression_refit_{tag}.png",
                   fits, args.depth_max, args.box_deg)
    fig_map(df, out_dir / "figures" / f"glodap_refit_boxes_map_{tag}.png", args.depth_max)


if __name__ == "__main__":
    main()
