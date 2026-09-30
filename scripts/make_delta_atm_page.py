"""Assemble the Delta_atm comparison page.

    python3 scripts/make_delta_atm_page.py --out <file.html>

Reads results/delta_atm_comparison.json and the two map PNGs, and writes a
single self-contained HTML file: maps embedded as data URIs, time series as
inline SVG drawn client-side from the embedded numbers.

Colour assignment is FIXED per variant and shared by every chart on the page,
so a variant keeps its identity throughout. The six hues are slots 1-6 of the
validated categorical palette, in order; Graven is a neutral dashed reference
rather than a seventh hue, because it is not one of the candidates.
"""

import argparse
import base64
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# slot: (light, dark). Order is the palette's; identity is the variant's.
SERIES = [
    ("krige",     "krige",      "kriged station field (incumbent)",       "#2a78d6", "#3987e5"),
    ("current",   "current",    "Miller NH/SH +10‰ 10°N→40°S", "#eb6834", "#d95926"),
    ("basu_band", "basu_band",  "Miller variant, +10‰ 20°N→20°S", "#1baf7a", "#199e70"),
    ("hua_geom",  "hua_geom",   "Hua step at Equator, our stations",      "#eda100", "#c98500"),
    ("hua_shape", "hua_shape",  "Hua's contrast, our level",              "#e87ba4", "#d55181"),
    ("hua_pub",   "hua_pub",    "Hua et al. (2021) as published",         "#008300", "#008300"),
]

LOO = [
    ("universal kriging (shipped)", 2.918, 2.916, "incumbent"),
    ("Hua step at Equator, our stations", 2.918, 2.918, "candidate"),
    ("Hua step, blended ±5°", 2.917, 2.916, "candidate"),
    ("drift only, sin(lat)¹", 2.921, 2.920, ""),
    ("Graven 3 bands 30°N/30°S", 3.172, 3.171, "candidate"),
    ("latitude-band mean (±20°)", 3.394, 3.367, ""),
    ("global mean — what TM5 gets today", 3.570, 3.570, "incumbent-tm5"),
    ("Miller NH/SH +10‰ — what ocedis uses today", 3.847, 3.699, "incumbent-ocedis"),
    ("Basu NH/SH +10‰", 4.365, 3.942, "candidate"),
]
LOO_EXTERNAL = [
    ("Hua et al. (2021), as published", 7.120, 2.971),
    ("Graven et al. (2017) CMIP6, as published", 3.263, 2.706),
]


def data_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()

    res = REPO_ROOT / "results"
    payload = json.loads((res / "delta_atm_comparison.json").read_text())
    figs = res / "figures"

    charts = {
        "years": payload["years"],
        "map_year": payload["map_year"],
        "series": [{"key": k, "name": n, "desc": d, "light": cl, "dark": cd}
                   for k, n, d, cl, cd in SERIES],
        "datm": {k: [r["global"] for r in payload["variants"][k]] for k, *_ in SERIES},
        "isoflux": {k: [r["isoflux"] for r in payload["variants"][k]] for k, *_ in SERIES},
        "gradient": {
            "ours": [r["nh"] - r["sh"] for r in payload["variants"]["hua_geom"]],
            "hua": [r["nh"] - r["sh"] for r in payload["hua"]],
            "graven": [[g["year"], g["nh"] - g["sh"]] for g in payload["graven"]
                       if payload["years"][0] <= g["year"] <= payload["years"][-1]],
        },
        "graven_global": [[g["year"], g["global"]] for g in payload["graven"]
                          if payload["years"][0] <= g["year"] <= payload["years"][-1]],
        "loo": LOO,
        "loo_external": LOO_EXTERNAL,
    }

    tpl = (REPO_ROOT / "scripts" / "delta_atm_page_template.html").read_text()
    html = tpl.replace("__DATA__", json.dumps(charts))
    html = html.replace("__MAP_DATM__", data_uri(figs / "d14c_maps_datm.png"))
    html = html.replace("__MAP_ISO__", data_uri(figs / "d14c_maps_isoflux_diff.png"))
    Path(args.out).write_text(html)
    kb = len(html) / 1024
    print(f"wrote {args.out}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
