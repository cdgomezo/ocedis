"""Maps of every Delta_atm candidate, and of what each one does to the isoflux.

    python3 scripts/plot_delta_atm_maps.py

Reads results/delta_atm_maps_2012.npz (written by
scripts/export_delta_atm_comparison.py) and writes two PNGs into
results/figures/.

Two maps, two colour jobs:

  Delta_atm itself        -> SEQUENTIAL. One hue, light to dark. The question
                             is magnitude, and there is no meaningful zero.
  isoflux minus krige     -> DIVERGING. Two hues about a neutral grey, limits
                             symmetric about zero. The question is sign: where
                             does this choice add ocean 14C uptake and where
                             does it remove it.

Both are drawn on a TRANSPARENT background with mid-grey annotation, so a
single file reads correctly on the light and the dark page surface. Baking a
white panel would make the dark theme unusable and doubling the payload with
a second render is not worth it when grey clears 3:1 against both.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Sequential ramp: the blue hue, steps 100..700, light -> dark.
SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf",
    "#184f95", "#0d366b",
])
# Diverging: blue <-> red about a neutral grey. Equal steps per arm; the red
# arm is stepped to the blue arm's lightness so neither pole dominates.
DIV = LinearSegmentedColormap.from_list("div_bluered", [
    "#0d366b", "#184f95", "#2a78d6", "#6da7ec", "#cde2fb",
    "#f0efec",
    "#fbd5d4", "#f4a9a8", "#e34948", "#c93534", "#8f2423",
])

INK = "#767676"          # clears 3:1 on both the light and dark page surface
LAND = "#9a9a9a"

LABELS = {
    "krige": "krige — kriged station field (incumbent)",
    "current": "current — Miller NH/SH +10‰ 10°N→40°S",
    "basu_band": "basu_band — Miller variant, +10‰ 20°N→20°S",
    "hua_geom": "hua_geom — Hua step at Equator, our stations",
    "hua_shape": "hua_shape — Hua's contrast, our level",
    "hua_pub": "hua_pub — Hua et al. (2021) as published",
}
ORDER = ["krige", "current", "basu_band", "hua_geom", "hua_shape", "hua_pub"]


def style(ax):
    ax.add_feature(cfeature.LAND, facecolor=LAND, edgecolor="none", zorder=2)
    ax.coastlines(linewidth=0.35, color=INK, zorder=3)
    ax.spines["geo"].set_edgecolor(INK)
    ax.spines["geo"].set_linewidth(0.5)


def grid_figure(z, lat, lon, titles, notes, cmap, norm_kw, cbar_label, out_path,
                suptitle, subtitle):
    """notes: one short line per panel, directly under its title.

    The panel note is not decoration. Several candidates differ only in the
    SIZE of a step, which a shared colour scale cannot show -- hua_geom and
    hua_shape are visually identical maps carrying a 6.3 vs 5.1 permil
    contrast. Without a number on the panel a reader would reasonably conclude
    they are the same field.
    """
    n = len(z)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.4 * ncol, 2.28 * nrow + 1.45),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=0)},
    )
    axes = np.atleast_1d(axes).ravel()

    mesh = None
    for ax, title, note, field in zip(axes, titles, notes, z):
        mesh = ax.pcolormesh(lon, lat, field, transform=ccrs.PlateCarree(),
                             cmap=cmap, shading="auto", zorder=1, **norm_kw)
        style(ax)
        ax.set_global()
        ax.set_title(f"{title}\n{note}", fontsize=8.4, color=INK, pad=3.5,
                     linespacing=1.5)
    for ax in axes[n:]:
        ax.set_visible(False)

    cax = fig.add_axes([0.31, 0.062, 0.38, 0.019])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label(cbar_label, fontsize=8.6, color=INK)
    cb.ax.tick_params(labelsize=7.6, colors=INK, length=2.5, width=0.5)
    cb.outline.set_edgecolor(INK)
    cb.outline.set_linewidth(0.5)

    fig.suptitle(suptitle, fontsize=12.5, color=INK, y=0.985)
    fig.text(0.5, 0.938, subtitle, ha="center", fontsize=8.8, color=INK)
    fig.subplots_adjust(top=0.855, bottom=0.125, hspace=0.30, wspace=0.05)
    fig.savefig(out_path, dpi=155, transparent=True)
    plt.close(fig)
    print(f"wrote {out_path}")


def zonal_stats(field, lat):
    """(area-weighted global mean, NH-SH contrast) of a (lat, lon) field."""
    prof = np.nanmean(field, axis=1)
    w = np.cos(np.radians(lat))
    gm = float(np.nansum(w * prof) / np.nansum(w))
    nh = lat >= 0
    contrast = (float(np.nansum(w[nh] * prof[nh]) / np.nansum(w[nh]))
                - float(np.nansum(w[~nh] * prof[~nh]) / np.nansum(w[~nh])))
    return gm, contrast


def main():
    src = REPO_ROOT / "results" / "delta_atm_maps_2012.npz"
    if not src.exists():
        raise SystemExit(f"{src} not found -- run "
                         "scripts/export_delta_atm_comparison.py first")
    d = np.load(src)
    lat, lon = d["lat"], d["lon"]
    out = REPO_ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    # --- Delta_atm itself, shared sequential scale ------------------------
    fields = [d[f"datm_{k}"] for k in ORDER]
    vmin = min(np.nanmin(f) for f in fields)
    vmax = max(np.nanmax(f) for f in fields)
    notes = []
    for k, f in zip(ORDER, fields):
        gm, contrast = zonal_stats(f, lat)
        notes.append(f"global {gm:.1f}‰   ·   NH−SH {contrast:+.1f}‰")
    grid_figure(
        fields, lat, lon, [LABELS[k] for k in ORDER], notes, SEQ,
        {"vmin": vmin, "vmax": vmax}, "Δ¹⁴C (‰), 2012 annual mean",
        out / "d14c_maps_datm.png",
        "The six candidate Δ_atm fields",
        "Only krige varies with longitude. Every other candidate is a set of "
        "latitude bands — the visible edges are band boundaries, not physics.",
    )

    # --- what it does to the isoflux, against the incumbent ---------------
    ref = d["isoflux_krige"]
    diffs, titles, notes = [], [], []
    for k in ORDER:
        if k == "krige":
            continue
        diff = d[f"isoflux_{k}"] - ref
        diffs.append(diff)
        titles.append(LABELS[k])
        notes.append(f"global {np.nansum(diff):+.0f} PgC‰/yr vs krige")
    lim = float(np.nanpercentile(np.abs(np.concatenate([x.ravel() for x in diffs])), 99.5))
    grid_figure(
        diffs, lat, lon, titles, notes, DIV,
        {"norm": TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)},
        "Δ ocean isoflux vs krige (PgC‰/yr per cell), 2012",
        out / "d14c_maps_isoflux_diff.png",
        "Where the Δ_atm choice actually changes the ocean isoflux",
        "Red = less negative than the incumbent (weaker ocean ¹⁴C sink); "
        "blue = more negative. Land is masked.",
    )


if __name__ == "__main__":
    main()
