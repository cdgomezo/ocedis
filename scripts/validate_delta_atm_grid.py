"""Validate the gridded atmospheric Delta14C field, and produce the figures.

    python3 scripts/validate_delta_atm_grid.py

Three checks, in decreasing order of how much they should be trusted:

1. Leave-one-station-out cross-validation, against a ladder of simpler
   baselines (global mean, latitude-band mean, drift-only at several
   polynomial degrees). This is what establishes how much spatial structure
   the network actually supports -- and it is the check that showed residual
   kriging adds ~0.1%, i.e. essentially nothing.

   The ladder now also carries the published zonal geometries -- Hua et al.
   (2021)'s equatorial step, Miller et al. (2025)'s NH/SH + 10 permil band,
   Basu et al. (2016)'s variant, and Graven et al. (2017)'s three CMIP6
   sectors -- in two families:

     FITTED   band values taken from the retained stations each month, so the
              row measures the GEOMETRY and is directly comparable to the
              drift and kriging rows.
     EXTERNAL the published curve itself, using none of the retained data.
              Strictly harder, and not comparable to the fitted rows.

   Every row is reported twice: raw RMSE, and RMSE after removing each
   month's own mean residual. The de-levelled column isolates SHAPE from
   LEVEL, which matters because Hua sits ~6-8 permil above Graven and Miller
   in absolute terms while matching this network on the interhemispheric
   gradient.
2. Against Graven et al. (2020)'s three CMIP6 sectors over the overlap.
3. Interhemispheric gradient against raw station differences (NWR-SPO,
   JFJ-SPO), which is independent of any interpolation choice.
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4 as nc  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ocean14c import hua  # noqa: E402
from ocean14c.atmfield import build, station_values_at, zonal_band_mean  # noqa: E402
from ocean14c.atmosphere import (  # noqa: E402
    BAND_OFFSETS, delta_atm_zonal, fit_bands,
)
from ocean14c.curvefit import month_midpoints  # noqa: E402
from ocean14c.krige import (  # noqa: E402
    DEFAULT_L_LAT_KM, DEFAULT_L_LON_KM, DEFAULT_OBS_ERR, DEFAULT_SILL,
    drift_basis, krige_points,
)

GRAVEN = REPO_ROOT.parent / (
    "Delta_atm/Delta14co2-in-air_input4MIPs_atmosphericState_C4MIP_"
    "ImperialCollege-2-0_gz-NHTRSH_1850-2015.nc"
)
C_OURS, C_REF, C_ALT = "#0072B2", "#E69F00", "#009E73"


def month_sets(stations, fits, times, obs_err=DEFAULT_OBS_ERR, min_n=6):
    out = []
    for t in times:
        la, lo, va, er = station_values_at(stations, fits, t, obs_err=obs_err)
        if len(va) >= min_n:
            out.append((t, la, lo, va, er))
    return out


def loo(data, predictor):
    """Leave-one-station-out over `data` -> (rmse, rmse_delevelled, per_point).

    The second metric subtracts each month's own mean residual before
    squaring. For predictors fitted to the retained stations this changes
    almost nothing -- their residuals are already near zero-mean. For an
    external published curve it removes the whole absolute-level offset,
    leaving only how well the SHAPE matches. Comparing the two columns is how
    "is Hua's level wrong, or its shape?" gets an answer.
    """
    residuals, per_point = [], []
    for t, lat, lon, val, err in data:
        n = len(val)
        month = []
        for i in range(n):
            k = np.arange(n) != i
            p = predictor(t, lat[k], lon[k], val[k], err[k],
                          float(lat[i]), float(lon[i]))
            if np.isfinite(p):
                month.append(p - val[i])
                per_point.append((lat[i], lon[i], val[i], p))
        if month:
            residuals.append(np.array(month))
    if not residuals:
        return np.nan, np.nan, per_point
    flat = np.concatenate(residuals)
    delev = np.concatenate([r - r.mean() for r in residuals])
    return (float(np.sqrt(np.mean(flat**2))),
            float(np.sqrt(np.mean(delev**2))),
            per_point)


def _drift_only(t, la, lo, v, e, tla, tlo, deg=1):
    F = drift_basis(la, deg)
    c, *_ = np.linalg.lstsq(F, v, rcond=None)
    return float((drift_basis(np.array([tla]), deg) @ c).ravel()[0])


def _krige(t, la, lo, v, e, tla, tlo, **kw):
    return float(krige_points(la, lo, v, e, tla, tlo, **kw)[0][0])


def _zonal_fitted(t, la, lo, v, e, tla, tlo, geometry="hemispheric", blend_deg=0.0,
                  fit_outside_offset_band=False):
    """Fit the geometry's band values to the retained stations, then evaluate.

    fit_outside_offset_band matters for the two +10 permil geometries. Miller
    et al. derive their hemispheric values from extratropical sites and then
    ADD 10 permil over the tropics; fitting NH/SH from all stations -- tropical
    ones included -- and then adding 10 on top double-counts the very gradient
    the band is meant to represent, which would score the band unfairly. With
    this flag the hemispheric values come only from stations outside the
    offset band, matching how the published construction actually works.
    """
    fit_lat, fit_val = la, v
    offset = BAND_OFFSETS.get(geometry)
    if fit_outside_offset_band and offset is not None:
        outside = ~((la <= offset["band_north"]) & (la >= offset["band_south"]))
        if outside.sum() >= 2 and len(set(la[outside] >= 0)) == 2:
            fit_lat, fit_val = la[outside], v[outside]
    bands = fit_bands(fit_lat, fit_val, geometry)
    return float(np.atleast_1d(
        delta_atm_zonal(np.array([tla]), bands, geometry, blend_deg=blend_deg)
    )[0])


def _hua_published(t, la, lo, v, e, tla, tlo, series=None, blend_deg=0.0):
    return float(hua.field_at(np.array([tla]), t, series=series,
                              blend_deg=blend_deg)[0])


def _graven_published(t, la, lo, v, e, tla, tlo, table=None):
    """Graven CMIP6 sector value for this latitude at the nearest annual
    midpoint. Annual data, so a decimal date within +-0.5 yr is the best
    available -- which is itself part of what the row measures."""
    gt, gd = table
    j = int(np.argmin(np.abs(gt - t)))
    if abs(gt[j] - t) > 0.75:
        return np.nan
    idx = 0 if tla >= 30 else (2 if tla <= -30 else 1)
    return float(gd[j, idx])


def load_graven():
    """(annual midpoint times, (n, 3) sector values) or None."""
    if not GRAVEN.exists():
        return None
    ds = nc.Dataset(GRAVEN)
    return 1850 + np.asarray(ds.variables["time"][:]), np.asarray(
        ds.variables["Delta14co2_in_air"][:]
    )


def fig_baselines(rows, out_path):
    """rows: (name, rmse, rmse_delevelled). Both bars are drawn -- the gap
    between them IS the absolute-level error, which for the external
    compilations is most of the story."""
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(rows) + 1.6))
    names = [r[0] for r in rows]
    raw = [r[1] for r in rows]
    delev = [r[2] for r in rows]
    y = np.arange(len(rows))
    ax.barh(y - 0.2, raw, height=0.38, color=C_OURS, edgecolor="k",
            linewidth=0.3, label="RMSE")
    ax.barh(y + 0.2, delev, height=0.38, color=C_REF, edgecolor="k",
            linewidth=0.3, label="de-levelled (shape only)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Leave-one-station-out RMSE (‰)")
    ax.set_title("How much spatial structure does the network support?")
    for i, (a, b) in enumerate(zip(raw, delev)):
        if np.isfinite(a):
            ax.text(a + 0.03, i - 0.2, f"{a:.2f}", va="center", fontsize=7)
        if np.isfinite(b):
            ax.text(b + 0.03, i + 0.2, f"{b:.2f}", va="center", fontsize=7)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_stations(result, out_path):
    st = result["stations"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    for code, s in st.items():
        c = C_OURS if s.is_background else C_REF
        ax1.scatter(s.lon, s.lat, c=c, s=45, edgecolor="k", linewidth=0.4, zorder=3)
        ax1.annotate(code, (s.lon, s.lat), fontsize=7,
                     xytext=(3, 3), textcoords="offset points")
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)
    ax1.set_xlabel("Longitude (°E)")
    ax1.set_ylabel("Latitude (°N)")
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Station network (blue = background, orange = fossil-affected)")

    order = sorted(st.values(), key=lambda s: s.lat)
    for i, s in enumerate(order):
        c = C_OURS if s.is_background else C_REF
        ax2.plot([s.start, s.end], [i, i], color=c, lw=3, solid_capstyle="butt")
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([f"{s.code} ({s.lat:+.0f}°)" for s in order], fontsize=7)
    ax2.set_xlabel("Year")
    ax2.set_title("Record coverage")
    ax2.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_graven(result, out_path):
    if not GRAVEN.exists():
        print(f"  Graven file not found at {GRAVEN}; skipping comparison figure")
        return None
    ds = nc.Dataset(GRAVEN)
    gt = 1850 + ds.variables["time"][:]
    gd = ds.variables["Delta14co2_in_air"][:]

    bands = [("NH (30–90°N)", 30, 90, 0), ("Tropics (30°S–30°N)", -30, 30, 1),
             ("SH (90–30°S)", -90, -30, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    stats = []
    for ax, (label, s, n, idx) in zip(axes, bands):
        ours = zonal_band_mean(result, s, n)
        yrs = np.unique(result["years"])
        ann = np.array([np.nanmean(ours[result["years"] == y]) for y in yrs])
        ax.plot(yrs, ann, color=C_OURS, marker="o", ms=3, label="this work")
        gsel = (gt >= yrs.min()) & (gt <= yrs.max())
        ax.plot(gt[gsel], gd[gsel, idx], color=C_REF, ls="--", marker="s",
                ms=3, label="Graven et al. (2020)")
        ax.set_title(label)
        ax.set_xlabel("Year")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        # Graven's time axis is annual MIDPOINTS (1850.5 ... 2015.5), so
        # calendar year y matches gt == y + 0.5. Matching to the nearest gt
        # instead silently compares against year y-1, which at ~5 permil/yr
        # of decline shows up as a spurious ~-6 permil bias.
        d = []
        for y in yrs:
            j = np.argmin(np.abs(gt - (y + 0.5)))
            if abs(gt[j] - (y + 0.5)) > 0.1:
                continue
            d.append(ann[list(yrs).index(y)] - gd[j, idx])
        if d:
            stats.append((label, float(np.mean(d)),
                          float(np.sqrt(np.mean(np.square(d)))), len(d)))
    axes[0].set_ylabel("Δ¹⁴C (‰)")
    fig.suptitle("Zonal sectors vs the Graven et al. (2020) CMIP6 forcing")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")
    return stats


def fig_field(result, out_path, sample_years=(1995, 2005, 2015, 2023)):
    avail = [y for y in sample_years if (result["years"] == y).any()]
    fig, axes = plt.subplots(len(avail), 2, figsize=(12, 3.0 * len(avail)))
    if len(avail) == 1:
        axes = axes[np.newaxis, :]
    lat, lon = result["lat"], result["lon"]
    for i, y in enumerate(avail):
        sel = (result["years"] == y) & (result["months"] == 7)
        f = result["field"][sel][0]
        s = result["sd"][sel][0]
        m1 = axes[i, 0].pcolormesh(lon, lat, f, cmap="viridis", shading="auto")
        axes[i, 0].set_title(f"{y}-07  Δ¹⁴C (‰)", fontsize=10)
        fig.colorbar(m1, ax=axes[i, 0])
        m2 = axes[i, 1].pcolormesh(lon, lat, s, cmap="magma", shading="auto")
        axes[i, 1].set_title(f"{y}-07  kriging σ (‰)", fontsize=10)
        fig.colorbar(m2, ax=axes[i, 1])
        for ax in axes[i]:
            ax.set_ylabel("lat")
    axes[-1, 0].set_xlabel("lon")
    axes[-1, 1].set_xlabel("lon")
    fig.suptitle("Gridded field and its uncertainty")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_loo_scatter(per_point, rmse, out_path):
    a = np.array(per_point)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    sc = ax.scatter(a[:, 2], a[:, 3], c=a[:, 0], cmap="coolwarm", s=12,
                    edgecolor="k", linewidth=0.2, vmin=-90, vmax=90)
    lim = [a[:, 2:4].min() - 5, a[:, 2:4].max() + 5]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Observed (fitted curve) Δ¹⁴C (‰)")
    ax.set_ylabel("Predicted with station withheld (‰)")
    ax.set_title(f"Leave-one-station-out\nRMSE = {rmse:.2f} ‰, n = {len(a)}")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="Latitude (°N)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=1990)
    p.add_argument("--end", type=int, default=2023)
    p.add_argument("--res", type=float, default=2.0)
    p.add_argument("--stride", type=int, default=6,
                   help="use every Nth month for cross-validation (speed)")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    out = Path(args.out_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)

    result = build(str(REPO_ROOT / "data/raw/Delta_atm"), args.start, args.end,
                   res_deg=args.res, include_regional=True)

    # Cross-validation uses background stations only -- the question is how
    # well the background field predicts itself.
    bg = {c: f for c, f in result["fits"].items()
          if result["stations"][c].is_background}
    times = month_midpoints(args.start, args.end)[:: args.stride]
    data = month_sets(result["stations"], bg, times)

    print(f"\ncross-validation: {len(data)} months, "
          f"{np.mean([len(d[2]) for d in data]):.1f} stations/month\n")

    def band_mean(t, la, lo, v, e, a, b):
        sel = np.abs(la - a) <= 20
        return float(v[sel].mean()) if sel.any() else float(v.mean())

    # ---- fitted from the retained stations each month --------------------
    rows = []
    rows.append(("global mean", *loo(data, lambda *x: float(x[3].mean()))[:2]))
    rows.append(("latitude-band mean (±20°)", *loo(data, band_mean)[:2]))
    for deg in (1, 2, 3):
        rows.append((f"drift only, sin(lat)^{deg}",
                     *loo(data, lambda *x, d=deg: _drift_only(*x, deg=d))[:2]))

    rmse_k, delev_k, per_point = loo(data, lambda *x: _krige(
        *x, l_lat_km=DEFAULT_L_LAT_KM, l_lon_km=DEFAULT_L_LON_KM, sill=DEFAULT_SILL))
    rows.append(("universal kriging (shipped)", rmse_k, delev_k))

    # The published geometries, with band values taken from the same retained
    # stations -- so these rows measure the geometry, not the compilation.
    rows.append(("Hua step at Equator (fitted)",
                 *loo(data, lambda *x: _zonal_fitted(*x, geometry="hemispheric"))[:2]))
    for bd in (5.0, 10.0, 20.0):
        rows.append((f"Hua step, blended ±{bd:.0f}° (fitted)",
                     *loo(data, lambda *x, b=bd: _zonal_fitted(
                         *x, geometry="hemispheric", blend_deg=b))[:2]))
    for g, label in (("miller_band", "Miller NH/SH +10‰ 10N→40S"),
                     ("basu_band", "Basu NH/SH +10‰ 20N→20S")):
        rows.append((f"{label} (fitted)",
                     *loo(data, lambda *x, gg=g: _zonal_fitted(*x, geometry=gg))[:2]))
        rows.append((f"{label} (fitted outside band)",
                     *loo(data, lambda *x, gg=g: _zonal_fitted(
                         *x, geometry=gg, fit_outside_offset_band=True))[:2]))
    rows.append(("Graven 3 bands 30N/30S (fitted)",
                 *loo(data, lambda *x: _zonal_fitted(*x, geometry="graven"))[:2]))

    print("\nFITTED -- band values from the retained stations. Comparable to")
    print("each other and to the drift/kriging rows.\n")
    print(f"  {'predictor':38s} {'RMSE':>7s} {'de-levelled':>12s}")
    for name, raw, delev in rows:
        print(f"  {name:38s} {raw:7.3f} {delev:12.3f}")

    # ---- external: the published curves, using none of our data ----------
    ext = []
    hua_series = None
    try:
        hua_series = hua.hemispheric_series()
    except FileNotFoundError as exc:
        print(f"\n  Hua curves unavailable ({exc}); skipping external rows")
    if hua_series is not None:
        ext.append(("Hua et al. (2021), as published",
                    *loo(data, lambda *x: _hua_published(*x, series=hua_series))[:2]))
        ext.append(("Hua et al. (2021), blended ±10°",
                    *loo(data, lambda *x: _hua_published(
                        *x, series=hua_series, blend_deg=10.0))[:2]))
    graven = load_graven()
    if graven is not None:
        ext.append(("Graven et al. (2017) CMIP6, as published",
                    *loo(data, lambda *x: _graven_published(*x, table=graven))[:2]))

    if ext:
        print("\nEXTERNAL -- the published curve itself, fitted to none of our")
        print("stations. Strictly harder than the rows above; NOT comparable to")
        print("them. The de-levelled column is the meaningful one, and it is")
        print("what separates a wrong LEVEL from a wrong SHAPE.\n")
        print(f"  {'predictor':38s} {'RMSE':>7s} {'de-levelled':>12s}")
        for name, raw, delev in ext:
            print(f"  {name:38s} {raw:7.3f} {delev:12.3f}")
        print("\n  Caveat: Hua's modern NH curve is built partly from stations")
        print("  scored here (JFJ, NWR/Niwot, BRW, BHD, CGO), so its external")
        print("  rows are not fully independent of the target.")

    fig_baselines(rows + ext, out / "d14c_loo_baselines.png")
    fig_loo_scatter(per_point, rmse_k, out / "d14c_loo_scatter.png")
    fig_stations(result, out / "d14c_stations.png")
    fig_field(result, out / "d14c_field.png")
    stats = fig_graven(result, out / "d14c_vs_graven.png")

    if stats:
        print("\nvs Graven et al. (2020), annual sector means")
        for label, bias, rmse, n in stats:
            print(f"  {label:22s} bias {bias:+6.2f}  RMSE {rmse:5.2f} permil  (n={n})")

    # Interhemispheric gradient, independent of interpolation
    nh = zonal_band_mean(result, 30, 90)
    sh = zonal_band_mean(result, -90, -30)
    grad = np.nanmean((nh - sh)[result["years"] >= 2010])
    print(f"\ninterhemispheric gradient (NH-SH, 2010+): {grad:+.2f} permil")


if __name__ == "__main__":
    main()
