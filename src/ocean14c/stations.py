"""Station registry for the atmospheric Delta14C network.

Every `.c14` file carries its own coordinates in the header (LATITUDE,
LONGITUDE, ALTITUDE), so the registry is parsed from the files rather than
hand-maintained -- adding a station means dropping its file into
data/raw/Delta_atm/.

Two things need judgement and are encoded here rather than discovered:

  * SITE_CLASS -- whether a site samples well-mixed background air. This
    matters because the gridded product is used as a TM5 forcing, and TM5
    already simulates fossil dilution via CO2_ff x fossil_del = -1.0.
    Prescribing a Delta_atm that also carries local fossil depletion would
    count that dilution twice. See docs/SCIENCE.md.

  * DUPLICATE FILES -- ALT and BHD each ship two files covering different
    periods/sampling; they are merged into one record per station code.
"""

import re
from pathlib import Path

from ocean14c.atm_stations import load_c14_station, load_drp_curl_event

# Site classification. "background" = remote/marine/high-altitude sites that
# sample well-mixed air most of the time. "regional" = sites with a known,
# substantial local fossil influence, which the background filter will reject
# heavily and which are downweighted (or excluded) when building the field.
#
# JFJ and MHD are deliberately "background": both are standard background
# stations (3450 m alpine, and Atlantic-facing marine air) whose occasional
# polluted samples are better handled by the statistical filter in
# background.py than by excluding the site wholesale -- they are also the two
# longest, best-sampled European records.
SITE_CLASS = {
    "MWO": "regional",   # Mt Wilson, above the Los Angeles basin
    "CMA": "regional",   # offshore Cape May, New Jersey -- US eastern seaboard outflow
    "NHA": "regional",   # offshore Portsmouth, New Hampshire -- same
}
DEFAULT_SITE_CLASS = "background"

_HEADER_RE = re.compile(r"^#\s*([A-Z][A-Z0-9/ _-]*?)\s*:\s*(.*)$")


def _parse_header(path):
    """Metadata dict from a .c14-style header (lines starting with '#')."""
    meta = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = _HEADER_RE.match(line.rstrip("\n"))
            if m:
                meta[m.group(1).strip()] = m.group(2).strip()
    return meta


def _coord(text):
    """'53.3266 N' or '-9.9044 E' -> float. The files always give signed
    values with a trailing hemisphere letter that is NOT a sign indicator
    (e.g. '-41.4100 N' for Baring Head, which is in the south), so the letter
    is discarded and the sign taken from the number."""
    return float(text.split()[0])


class Station:
    """One station's metadata and observations."""

    def __init__(self, code, name, lat, lon, alt, series, files, site_class):
        self.code = code
        self.name = name
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.series = series          # sorted [(decimal_date, delta14C_permil), ...]
        self.files = files
        self.site_class = site_class

    @property
    def n(self):
        return len(self.series)

    @property
    def start(self):
        return self.series[0][0] if self.series else None

    @property
    def end(self):
        return self.series[-1][0] if self.series else None

    @property
    def is_background(self):
        return self.site_class == "background"

    def __repr__(self):
        return (
            f"Station({self.code}, {self.lat:+.2f}, {self.lon:+.2f}, "
            f"n={self.n}, {self.start:.1f}-{self.end:.1f}, {self.site_class})"
        )


def load_stations(delta_atm_dir, include_regional=True):
    """Build the station registry from every readable file in `delta_atm_dir`.

    Files sharing a STATION CODE are merged (ALT and BHD each have two), with
    duplicate timestamps dropped -- keeping the first occurrence, which is the
    longer/primary record because files are visited in sorted order.

    The ObsPack-format `*_curl_event.txt` files are skipped: they duplicate
    stations (NWR, DRP) already covered by the .c14 files, in a format with no
    coordinate header. `ocean14c.atm_stations.load_drp_curl_event` remains
    available for the Ocedis pipeline, which uses the DRP one directly.

    Returns {code: Station}, sorted by code.
    """
    delta_atm_dir = Path(delta_atm_dir)
    paths = sorted(p for p in delta_atm_dir.iterdir() if p.is_file())

    merged = {}
    for path in paths:
        if path.name.endswith("_curl_event.txt"):
            continue
        meta = _parse_header(path)
        code = meta.get("STATION CODE")
        if not code or "LATITUDE" not in meta or "LONGITUDE" not in meta:
            continue
        try:
            series = load_c14_station(path)
        except (StopIteration, ValueError, IndexError):
            continue
        if not series:
            continue

        if code not in merged:
            merged[code] = {
                "name": meta.get("STATION NAME", code),
                "lat": _coord(meta["LATITUDE"]),
                "lon": _coord(meta["LONGITUDE"]),
                "alt": _coord(meta.get("ALTITUDE", "0")),
                "obs": {},
                "files": [],
            }
        entry = merged[code]
        entry["files"].append(path.name)
        for t, v in series:
            entry["obs"].setdefault(t, v)   # first file wins on duplicates

    stations = {}
    for code, e in merged.items():
        site_class = SITE_CLASS.get(code, DEFAULT_SITE_CLASS)
        if site_class == "regional" and not include_regional:
            continue
        stations[code] = Station(
            code=code,
            name=e["name"],
            lat=e["lat"],
            lon=e["lon"],
            alt=e["alt"],
            series=sorted(e["obs"].items()),
            files=sorted(e["files"]),
            site_class=site_class,
        )
    return dict(sorted(stations.items()))


def registry_table(stations):
    """Human-readable inventory, one line per station."""
    lines = [
        f"{'code':6s} {'lat':>8s} {'lon':>9s} {'alt':>8s} {'n':>6s} "
        f"{'start':>8s} {'end':>8s}  class       name"
    ]
    for s in stations.values():
        lines.append(
            f"{s.code:6s} {s.lat:+8.2f} {s.lon:+9.2f} {s.alt:8.0f} {s.n:6d} "
            f"{s.start:8.2f} {s.end:8.2f}  {s.site_class:10s}  {s.name[:30]}"
        )
    return "\n".join(lines)
