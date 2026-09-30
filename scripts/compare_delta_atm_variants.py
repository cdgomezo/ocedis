"""What each candidate Delta_atm does to the ocean disequilibrium isoflux.

    python3 scripts/compare_delta_atm_variants.py --start 2011 --end 2013

The leave-one-station-out ladder in validate_delta_atm_grid.py says which
representation predicts a withheld station best. That is necessary but not
sufficient: Delta_atm enters the disequilibrium fluxes as
Delta_source - Delta_atm, so what actually matters is what the choice does to
the isoflux. This script measures that.

Candidates:

  current       NH/SH + 10 permil over 10N..40S (Miller et al. 2025). What the
                ocedis pipeline uses today, and the regression guard: it must
                reproduce the shipped numbers exactly.
  basu_band     the same with Basu et al. (2016)'s 20N..20S band.
  hua_geom      plain NH/SH step at the Equator -- Hua et al. (2021)'s geometry
                for 1973 onward -- with band values from our own stations.
  hua_geom_b10  the same, blended over +-10 deg so the forcing is continuous.
  hua_pub       Hua's published curves, absolute.
  hua_shape     Hua's shape re-levelled onto our own global mean. Hua sits ~6-8
                permil above Graven and Miller in absolute terms while agreeing
                with our network on the interhemispheric gradient, so shape and
                level have to be separable.
  krige         the shipped 1x1 kriged field (output/d14c_atm.<year>.nc). The
                incumbent on measured skill, built from all 22 stations rather
                than the six the hemispheric splice uses -- and, today, NOT what
                ocedis consumes.

Also reports the implied terrestrial consequence. biodis is not built in this
tree (the LPJ product is external), so its Delta_atm is unknown and probably
different again; the number here is what a shift of this size would do to the
terrestrial isoflux, F_ba * dDelta_atm.
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ocean14c import hua  # noqa: E402
from ocean14c.atmosphere import area_weighted_mean  # noqa: E402
from ocean14c.pipeline import reconstruct_year  # noqa: E402

# Miller et al. (2025) Table 1, global heterotrophic respiration used for the
# terrestrial disequilibrium isoflux -- the lever arm on a Delta_atm shift.
F_BA_PGC_PER_YR = 54.5

VARIANTS = {
    "current":      dict(geometry="miller_band"),
    "basu_band":    dict(geometry="basu_band"),
    "hua_geom":     dict(geometry="hemispheric"),
    "hua_geom_b10": dict(geometry="hemispheric", blend_deg=10.0),
    "hua_pub":      dict(geometry="hua", level="native"),
    "hua_shape":    dict(geometry="hua", level="stations"),
    "krige":        dict(geometry="krige"),
}

# Okabe-Ito, the palette the rest of this repo's figures already use. Assigned
# in fixed order per variant so a variant keeps its colour across figures.
C_KRIGE, C_CUR = "#009E73", "#E69F00"
VARIANT_COLOUR = {
    "krige": C_KRIGE,
    "current": C_CUR,
    "basu_band": "#D55E00",
    "hua_geom": "#0072B2",
    "hua_geom_b10": "#56B4E9",
    "hua_pub": "#CC79A7",
    "hua_shape": "#332288",
}


def zonal_profile(result):
    """Annual-mean Delta_atm as a function of latitude [permil].

    Most variants are zonally uniform; `krige` is not, so its longitude axis
    is averaged away here. That averaging is only for the diagnostic -- the
    isoflux itself uses the full field.
    """
    da = result["delta_atm_monthly"].mean(dim="month")
    if "lon" in da.dims:
        da = da.mean(dim="lon")
    return np.asarray(da["lat"].values), np.asarray(da.values)


def run(years, data_dir):
    hua_series = hua.hemispheric_series()
    out = {}
    for name, spec in VARIANTS.items():
        rows = []
        for year in years:
            r = reconstruct_year(
                year, data_dir=data_dir, atm_variant=spec,
                krige_dir=str(REPO_ROOT / "output"), hua_series=hua_series,
            )
            lat, prof = zonal_profile(r)
            rows.append({
                "year": year,
                "global_isoflux": float(r["global_isoflux"]),
                "global_delta_odis": float(r["global_delta_odis"]),
                "socean_delta_odis": float(r["socean_delta_odis"]),
                "global_delta_atm": area_weighted_mean(lat, prof),
                "lat": lat,
                "profile": prof,
            })
        out[name] = rows
    return out


def report(results, years):
    ref = {r["year"]: r for r in results["current"]}

    print("\nGlobal ocean isoflux, PgC permil/yr (Table 1's aggregator)\n")
    head = f"  {'variant':14s}" + "".join(f"{y:>12d}" for y in years) + f"{'mean d':>12s}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name, rows in results.items():
        deltas = [r["global_isoflux"] - ref[r["year"]]["global_isoflux"] for r in rows]
        line = f"  {name:14s}" + "".join(f"{r['global_isoflux']:12.1f}" for r in rows)
        print(line + f"{np.mean(deltas):12.1f}")

    print("\nGlobal mean Delta_atm, permil  (and its offset from `current`)\n")
    print(f"  {'variant':14s}" + "".join(f"{y:>12d}" for y in years))
    for name, rows in results.items():
        print(f"  {name:14s}" + "".join(f"{r['global_delta_atm']:12.2f}" for r in rows))

    print("\nFlux-weighted Delta_odis, permil (global / Southern Ocean)\n")
    for name, rows in results.items():
        g = np.mean([r["global_delta_odis"] for r in rows])
        s = np.mean([r["socean_delta_odis"] for r in rows])
        print(f"  {name:14s} {g:8.2f} {s:10.2f}")

    print("\nImplied terrestrial consequence")
    print("  biodis is not built in this tree, so its Delta_atm is unknown and")
    print(f"  probably different again. At F_ba = {F_BA_PGC_PER_YR} PgC/yr a shift")
    print("  of dDelta_atm moves the terrestrial isoflux by F_ba * dDelta_atm:\n")
    for name, rows in results.items():
        if name == "current":
            continue
        d = np.mean([r["global_delta_atm"] - ref[r["year"]]["global_delta_atm"]
                     for r in rows])
        print(f"  {name:14s} dDelta_atm {d:+6.2f} permil  ->  "
              f"terdis {-d * F_BA_PGC_PER_YR:+8.0f} PgC permil/yr")


def station_annual_means(data_dir, years):
    """Each background station's own fitted annual-mean Delta14C, by latitude.

    This is the observational reference the candidates are trying to match --
    the same curves the leave-one-out ladder scores against, so the figure and
    the table are talking about the same thing.
    """
    from ocean14c.atmfield import station_values_at
    from ocean14c.background import filter_all
    from ocean14c.stations import load_stations

    stations = load_stations(str(Path(data_dir) / "raw" / "Delta_atm"))
    fits, _ = filter_all(stations)
    bg = {c: f for c, f in fits.items() if stations[c].is_background}

    out = {}
    for year in years:
        acc = {}
        for t in year + (np.arange(12) + 0.5) / 12.0:
            la, _, va, _ = station_values_at(stations, bg, float(t))
            for a, v in zip(la, va):
                acc.setdefault(float(a), []).append(float(v))
        out[year] = [(a, float(np.mean(v))) for a, v in acc.items() if len(v) >= 6]
    return out


def fig_profiles(results, years, out_path, data_dir=None):
    """Zonal profiles, with the kriged incumbent drawn on every panel and the
    station network's own annual means as the observational reference."""
    obs = station_annual_means(data_dir, years) if data_dir else {}
    n = len(years)
    fig, axes = plt.subplots(1, n, figsize=(4.9 * n, 5.0), sharey=True)
    axes = np.atleast_1d(axes)

    order = ["current", "basu_band", "hua_pub", "hua_shape",
             "hua_geom", "hua_geom_b10", "krige"]
    labels = {"current": "current (Miller +10‰ band)",
              "krige": "krige (incumbent)",
              "hua_geom": "Hua step, our stations",
              "hua_geom_b10": "Hua step, blended ±10°",
              "hua_pub": "Hua, as published",
              "hua_shape": "Hua shape, our level"}

    for ax, year in zip(axes, years):
        for name in order:
            r = next(x for x in results[name] if x["year"] == year)
            lw = 2.8 if name == "krige" else (2.0 if name == "current" else 1.2)
            ax.plot(r["profile"], r["lat"], color=VARIANT_COLOUR[name], lw=lw,
                    label=labels.get(name, name),
                    zorder=5 if name == "krige" else 3)
        for a, v in obs.get(year, []):
            ax.plot(v, a, "o", ms=5, mfc="none", mec="0.15", mew=1.1, zorder=6,
                    label="station annual mean" if a == obs[year][0][0] else None)
        ax.axhline(0, color="0.7", lw=0.6, zorder=1)
        ax.set_title(f"{year} annual mean")
        ax.set_xlabel("Δ¹⁴C (‰)")
        ax.set_ylim(-90, 90)
        ax.margins(x=0.06)
        ax.grid(True, alpha=0.22)
    axes[0].set_ylabel("Latitude (°N)")

    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab, frameon=False, fontsize=8.5, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Δ_atm candidates against the stations they are meant to represent")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_path}")


def fig_isoflux(results, years, out_path):
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ref = {r["year"]: r["global_isoflux"] for r in results["current"]}
    for name, rows in results.items():
        y = [r["global_isoflux"] - ref[r["year"]] for r in rows]
        ax.plot([r["year"] for r in rows], y, marker="o", ms=4, label=name,
                color=VARIANT_COLOUR[name],
                lw=2.6 if name == "krige" else 1.4)
    ax.axhline(0, color=C_CUR, lw=2.0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Δ global ocean isoflux vs current (PgC‰/yr)")
    ax.set_title("What the Δ_atm choice costs the ocean disequilibrium isoflux")
    ax.set_xticks(list(years))
    ax.grid(True, alpha=0.25)
    # Outside the axes: hua_geom and hua_geom_b10 differ by <1 PgC permil/yr
    # and overplot, so the legend is the only thing distinguishing them and it
    # must not sit on top of a line.
    ax.legend(frameon=False, fontsize=8.5, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=2011)
    p.add_argument("--end", type=int, default=2013)
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "figures"))
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    years = list(range(args.start, args.end + 1))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = run(years, args.data_dir)

    # Regression guard: `current` must be the shipped path, untouched.
    for year in years:
        got = next(r for r in results["current"] if r["year"] == year)["global_isoflux"]
        want = float(reconstruct_year(year, data_dir=args.data_dir)["global_isoflux"])
        assert got == want, (
            f"{year}: variant path {got} != default path {want} -- the refactor "
            "changed the shipped numbers"
        )
    print(f"regression guard: `current` reproduces the default path exactly "
          f"for {years[0]}-{years[-1]}")

    report(results, years)
    fig_profiles(results, years, out / "d14c_variant_profiles.png",
                 data_dir=args.data_dir)
    fig_isoflux(results, years, out / "d14c_variant_isoflux.png")


if __name__ == "__main__":
    main()
