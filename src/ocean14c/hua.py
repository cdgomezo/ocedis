"""Hua et al. (2021) compiled zonal atmospheric Delta14C.

Quan Hua et al. (2022) "Atmospheric radiocarbon for the period 1950-2019",
Radiocarbon 64(4), 723-745, doi:10.1017/RDC.2021.95. Data files and their
provenance: data/raw/Hua2021/SOURCE.md.

This module exists to make Hua a first-class, testable alternative to the
station-fitted field this repo builds itself (ocean14c.atmfield), so the two
can be scored against each other rather than argued about.

THREE THINGS THAT WILL BITE:

1. The files carry F14C, not Delta14C. Delta14C = (F14C - 1) * 1000.
   A curve read as permil is off by a factor of ~1000 and looks like zero.

2. For 1973 onward there are only TWO distinct curves, not five. Hua pools the
   records by hemisphere from 1973 (paper Table 1, "PERIOD 1973-2019"), so
   NHZ1 == NHZ2 == NHZ3 and SHZ3 == SHZ1_2. Measured over 2011-2013 the NH
   zones agree to 0.1 permil and the SH zones to 0.2. Anything that claims to
   resolve five zones in the modern era is resolving rounding noise. The five
   zones are only live before 1973, which is why zone_of() refuses that era
   rather than inventing tropical-low-pressure-belt positions the paper
   derives from NCEP/NCAR sea-level pressure and does not tabulate.

3. The post-2019 extrapolation in the paper's Appendix is an exponential IN
   F14C. Applying those coefficients to Delta14C gives nonsense by three
   orders of magnitude. See extrapolate_f14c().

   Worse, that exponential does not land on the record it extends. Evaluated
   at the last data point, 2019.375, it gives F14C 1.00664 (NH) and 1.01589
   (SH) against the tabulated 1.0084 and 1.0195 -- steps of 1.8 and 3.6
   permil. The fits are least-squares over 1993/1994-2019 and were never
   constrained to be continuous at the join. Used raw, they put a step into
   the forcing exactly where a 2012-2024 inversion crosses into 2020.
   extrapolate_f14c(anchor=True), the default everywhere in this module,
   rescales the exponential to match the last observation while keeping its
   decay rate.

Hua's ABSOLUTE LEVEL is an open question, not a settled fact. Over 2000-2013
these curves sit ~4-8 permil above Graven et al. (2017/2020) in the NH and
~7-10 above in the SH, while Miller et al. (2025) report agreeing with Graven
to 0.2 +- 1.3 permil. The interhemispheric GRADIENT, by contrast, matches this
repo's own station network well (Hua -5 to -9 permil for 2011-2013 vs our
-6.2). Hence `level="anomaly"`: take Hua's shape, keep somebody else's level.
"""

import os

import numpy as np

# Curve file stems, north to south.
NH_ZONES = ("NHZ1", "NHZ2", "NHZ3")
SH_ZONES = ("SHZ3", "SHZ1_2")
ZONES = NH_ZONES + SH_ZONES

# Records are pooled by hemisphere from this year on (paper Table 1).
POOLED_FROM_YEAR = 1973

# Last data point in the compilation, 2019.375 (Supplementary Tables S2a-e).
LAST_DATA_YEAR = 2019.375

# Paper Appendix: y = a * exp(b * t) fitted to the monthly F14C data, NH over
# 1993-2019 and SH over 1994-2019. Recommended for extrapolation but explicitly
# capped by the authors at "no more than 5 years after 2019".
EXTRAP_COEFFS = {
    "NH": (7112.837567, -0.004389),
    "SH": (4271.928748, -0.004132),
}
EXTRAP_MAX_YEAR = LAST_DATA_YEAR + 5.0


def default_data_dir():
    """data/raw/Hua2021 relative to the repo root."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "data", "raw", "Hua2021")


def load_zone(path):
    """One CALIBomb curve file -> (decimal_year, delta14c_permil, sigma_permil).

    The file is (year, F14C, 1-sigma F14C); both value columns are converted to
    permil here so nothing downstream has to remember.
    """
    t, f, s = [], [], []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                row = [float(p) for p in parts[:3]]
            except ValueError:
                continue          # header or comment
            t.append(row[0])
            f.append(row[1])
            s.append(row[2])
    if not t:
        raise ValueError(f"no numeric rows in {path}")
    t = np.array(t)
    return t, (np.array(f) - 1.0) * 1000.0, np.array(s) * 1000.0


def load_zonal_curves(data_dir=None):
    """All five zones -> {zone: (time, delta14c, sigma)}, permil."""
    data_dir = data_dir or default_data_dir()
    out = {}
    for z in ZONES:
        path = os.path.join(data_dir, f"{z}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- see data/raw/Hua2021/SOURCE.md for how to "
                f"fetch it (http://calib.org/CALIBomb/{z}.txt)"
            )
        out[z] = load_zone(path)
    return out


def zone_of(lat, lon=None, t=None):
    """Hua zone for a latitude (and, before 1973, longitude).

    From 1973 onward the only boundary that carries information is the
    Equator, so this returns "NH" or "SH" -- deliberately not one of the five
    zone names, because claiming to place a point in NHZ1 vs NHZ3 in the modern
    era would be claiming a distinction the data do not contain.

    Before 1973 the five zones do differ (by up to 570 permil at the bomb
    peak), and placing a point among them needs the mean JJA and DJF positions
    of the tropical low-pressure belt from NCEP/NCAR sea-level pressure. The
    paper draws them in its Figure 1 but tabulates nothing, so rather than
    invent a boundary this raises.
    """
    if t is not None and t < POOLED_FROM_YEAR:
        raise NotImplementedError(
            f"t={t} is before {POOLED_FROM_YEAR}, where Hua's five zones are "
            "genuinely distinct. Placing a point among them requires the mean "
            "JJA/DJF tropical low-pressure-belt positions, which the paper "
            "draws in Figure 1 but does not tabulate. Digitise Figure 1 or "
            "recompute the TLPB from NCEP/NCAR SLP before using this era."
        )
    return np.where(np.asarray(lat, dtype=float) >= 0.0, "NH", "SH")


def extrapolate_f14c(t, hemisphere, anchor=True, series=None, data_dir=None):
    """Paper Appendix exponential, y = a*exp(b*t), evaluated in F14C.

    Returns F14C, NOT permil -- the coefficients are fitted in F14C and the
    caller converting is the point.

    anchor=True (the default) rescales the curve by a constant factor so it
    passes through the last tabulated value, removing the 1.8 permil (NH) /
    3.6 permil (SH) step the raw fit has at 2019.375. The factor is
    multiplicative, so the decay rate -- the only thing the fit is really
    for -- is untouched. anchor=False reproduces the paper literally.

    Beyond EXTRAP_MAX_YEAR the paper says its own extrapolation is not usable;
    we raise rather than return a number that looks fine.
    """
    t = np.asarray(t, dtype=float)
    if np.any(t > EXTRAP_MAX_YEAR):
        raise ValueError(
            f"extrapolation past {EXTRAP_MAX_YEAR:.3f} requested. Hua et al. "
            "cap their own exponential at ~5 years beyond 2019; past that use "
            "the station network (ocean14c.atmfield), which runs to 2025."
        )
    a, b = EXTRAP_COEFFS[hemisphere]
    out = a * np.exp(b * t)
    if not anchor:
        return out

    series = series or hemispheric_series(data_dir)
    ts, ds = series[hemisphere]
    observed_f14c = 1.0 + ds[-1] / 1000.0
    return out * (observed_f14c / (a * np.exp(b * ts[-1])))


def hemispheric_series(data_dir=None, zones=None):
    """Monthly NH and SH Delta14C series -> {"NH": (t, d14c), "SH": (t, d14c)}.

    Post-1973 the zones within a hemisphere are identical to rounding, so the
    hemispheric series is their mean; pre-1973 it is an unweighted mean of the
    zones and should not be used (the zones differ by hundreds of permil).
    """
    curves = load_zonal_curves(data_dir) if zones is None else zones
    out = {}
    for name, members in (("NH", NH_ZONES), ("SH", SH_ZONES)):
        ts = [curves[z][0] for z in members]
        base = ts[0]
        if not all(len(x) == len(base) and np.allclose(x, base) for x in ts[1:]):
            raise ValueError(f"{name} zone time axes differ; cannot average")
        out[name] = (base, np.mean([curves[z][1] for z in members], axis=0))
    return out


def value_at(t, hemisphere, series=None, data_dir=None, anchor=True):
    """Delta14C [permil] for one hemisphere at decimal date(s) `t`.

    Linear interpolation within the record; the paper's exponential beyond it,
    anchored to the record end by default (see extrapolate_f14c). The curves
    are already monthly, so linear interpolation between adjacent points
    introduces nothing.
    """
    series = series or hemispheric_series(data_dir)
    ts, ds = series[hemisphere]
    t = np.atleast_1d(np.asarray(t, dtype=float))

    out = np.interp(t, ts, ds, left=np.nan, right=np.nan)
    past = t > ts[-1]
    if past.any():
        out[past] = (
            extrapolate_f14c(t[past], hemisphere, anchor=anchor, series=series)
            - 1.0
        ) * 1000.0
    return out


def field_at(lat, t, level="absolute", global_mean=None,
             series=None, data_dir=None, blend_deg=0.0):
    """Hua's Delta14C as a zonally uniform function of latitude, in permil.

    lat        : latitudes [degrees]
    t          : one decimal date
    level      : "absolute" -- Hua's own values.
                 "anomaly"  -- Hua's NH/SH contrast about its own area-weighted
                               global mean, re-centred on `global_mean`. This
                               separates the shape (where Hua agrees with our
                               station network) from the level (where it sits
                               ~6-8 permil above Graven and Miller).
    global_mean: required for level="anomaly", permil.
    blend_deg  : half-width of a linear blend across the Equator. 0 gives the
                 paper's hard step; a few degrees removes a discontinuity that
                 a transport model would otherwise see in the forcing.
    """
    lat = np.asarray(lat, dtype=float)
    nh = float(value_at(t, "NH", series=series, data_dir=data_dir)[0])
    sh = float(value_at(t, "SH", series=series, data_dir=data_dir)[0])

    if blend_deg > 0:
        w = np.clip((lat + blend_deg) / (2.0 * blend_deg), 0.0, 1.0)
    else:
        w = (lat >= 0.0).astype(float)
    field = sh + w * (nh - sh)

    if level == "absolute":
        return field
    if level == "anomaly":
        if global_mean is None:
            raise ValueError('level="anomaly" needs global_mean [permil]')
        return field - area_weighted_mean(lat, field) + float(global_mean)
    raise ValueError(f"unknown level: {level!r}")


def area_weighted_mean(lat, field):
    """cos(lat)-weighted mean of a zonally uniform field."""
    w = np.cos(np.radians(np.asarray(lat, dtype=float)))
    return float(np.sum(w * np.asarray(field, dtype=float)) / np.sum(w))
