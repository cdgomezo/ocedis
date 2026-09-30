"""Per-station curve fitting with NOAA's ccgcrv (Thoning et al. 1989).

This is the temporal half of the gridded Delta_atm construction: each station
record is fitted once here, giving a continuous curve that can be evaluated at
any month. The spatial half (ocean14c.krige) then works one month at a time on
the fitted station values. Keeping the two separate is what makes the spatial
step swappable for optimal interpolation later.

ccgcrv fits a polynomial + harmonics, then low-pass-filters the residuals with
an FFT and adds them back -- the filtered-residual step that
`ocean14c.atmosphere.fit_year_edge_values` approximates with a bare quadratic.
"""

import numpy as np

from ocean14c.vendor.ccg_filter import ccgFilter

# Defaults. Delta14C records are ~14-day integrated samples (~25/yr), far
# sparser than the daily CO2 data ccgcrv was designed around, so the harmonic
# count is reduced from NOAA's default of 4: with ~25 samples/yr, 4 harmonics
# (8 parameters) chases noise. 2 captures the annual and semiannual cycle.
SHORTTERM_DAYS = 80      # NOAA default
LONGTERM_DAYS = 667      # NOAA default
NUM_POLY = 3             # quadratic
NUM_HARMONICS = 2

MIN_POINTS = 30          # below this a curve fit is not meaningful
MIN_SPAN_YEARS = 3.0


class StationFit:
    """A fitted station curve plus the diagnostics needed to judge it."""

    def __init__(self, code, filt, t, v, rejected_t=None):
        self.code = code
        self.filt = filt
        self.t = np.asarray(t)
        self.v = np.asarray(v)
        self.rejected_t = np.asarray(rejected_t if rejected_t is not None else [])

    @property
    def start(self):
        return float(self.t.min())

    @property
    def end(self):
        return float(self.t.max())

    @property
    def n(self):
        return len(self.t)

    @property
    def n_rejected(self):
        return len(self.rejected_t)

    @property
    def residual_sd(self):
        """Scatter of the retained observations about the smooth curve."""
        return float(np.std(self.v - self.smooth(self.t)))

    def _eval(self, method, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.asarray(getattr(self.filt, method)(x), dtype=float)
        return np.atleast_1d(out)

    def smooth(self, x):
        """Function + short-term filtered residual (keeps the seasonal cycle)."""
        return self._eval("getSmoothValue", x)

    def trend(self, x):
        """Function + long-term filtered residual (deseasonalised)."""
        return self._eval("getTrendValue", x)

    def covers(self, x, margin=0.0):
        """Whether `x` lies inside the observed record (no extrapolation).

        ccgcrv is a smoother, not a forecaster: `getSmoothValue` returns NaN
        outside the record. Callers must either mask on this or use
        `extrapolate`/`evaluate`.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return (x >= self.start - margin) & (x <= self.end + margin)

    def harmonics(self, x):
        """Fitted seasonal cycle (annual + semiannual)."""
        return self._eval("getHarmonicValue", x)

    def growth_rate(self, x):
        """Derivative of the trend [permil/yr]."""
        return self._eval("getGrowthRateValue", x)

    def extrapolate(self, x):
        """Continue the curve beyond the record.

        Linear continuation of the deseasonalised trend at its end-of-record
        growth rate, plus the fitted seasonal cycle:

            value(t) = trend(t_end) + growth(t_end) * (t - t_end) + harmonics(t)

        Chosen over ccgcrv's own polynomial (`getFunctionValue`) on evidence.
        A hold-out test over 11 long records -- truncate 2 years early, predict
        the withheld 2 years -- gives:

            polynomial            mean RMSE 3.62 permil, worst 8.28
            linear trend          mean RMSE 2.89 permil, worst 4.03
            linear + harmonics    mean RMSE 2.78 permil, worst 4.25

        The quadratic term is what makes the polynomial unstable once it leaves
        the data (NMY +7.9 permil bias, CGO +5.5). Holding the growth rate
        fixed is both better behaved and physically the more conservative
        assumption. Adding the harmonics back matters at the seasonal NH sites
        that carry the extrapolated years (JFJ 3.52 -> 2.83, NWR 3.20 -> 2.93,
        MHD 3.25 -> 2.79).

        NOTE this assumes the Delta14C decline rate stays at its end-of-record
        value. The decline has in fact been accelerating as fossil dilution
        grows, so extrapolated values are likely a slight OVER-estimate of
        Delta14C (too positive).
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.full(x.shape, np.nan)
        for anchor, sel in ((self.end, x > self.end), (self.start, x < self.start)):
            if not sel.any():
                continue
            y0 = float(self.trend(anchor)[0])
            g = float(self.growth_rate(anchor)[0])
            out[sel] = y0 + g * (x[sel] - anchor) + self.harmonics(x[sel])
        return out

    def evaluate(self, x):
        """Smoothed value inside the record, extrapolated outside."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.full(x.shape, np.nan)
        inside = self.covers(x)
        if inside.any():
            out[inside] = self.smooth(x[inside])
        if (~inside).any():
            out[~inside] = self.extrapolate(x[~inside])
        return out

    def gap(self, x):
        """Years by which `x` falls outside the record (0.0 if inside)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.maximum.reduce([self.start - x, x - self.end,
                                  np.zeros_like(x)])

    def __repr__(self):
        return (
            f"StationFit({self.code}, n={self.n}, {self.start:.1f}-{self.end:.1f}, "
            f"sd={self.residual_sd:.2f}permil, rejected={self.n_rejected})"
        )


def fit_station(
    code,
    series,
    shortterm=SHORTTERM_DAYS,
    longterm=LONGTERM_DAYS,
    numpoly=NUM_POLY,
    numharm=NUM_HARMONICS,
    rejected_t=None,
):
    """Fit one station's (decimal_date, value) series with ccgcrv.

    Returns a StationFit, or None if the record is too short or too sparse to
    support a curve fit (see MIN_POINTS / MIN_SPAN_YEARS).
    """
    if not series:
        return None
    t = np.array([d for d, _ in series], dtype=float)
    v = np.array([x for _, x in series], dtype=float)
    order = np.argsort(t)
    t, v = t[order], v[order]

    if len(t) < MIN_POINTS or (t.max() - t.min()) < MIN_SPAN_YEARS:
        return None

    filt = ccgFilter(
        t, v,
        shortterm=shortterm,
        longterm=longterm,
        numpolyterms=numpoly,
        numharmonics=numharm,
    )
    return StationFit(code, filt, t, v, rejected_t=rejected_t)


def fit_all(stations, **kwargs):
    """Fit every station in a registry. Returns {code: StationFit}, skipping
    records that are too short."""
    fits = {}
    for code, st in stations.items():
        fit = fit_station(code, st.series, **kwargs)
        if fit is not None:
            fits[code] = fit
    return fits


def month_midpoints(year_start, year_end):
    """Decimal-date month midpoints, inclusive of both years."""
    years = np.arange(year_start, year_end + 1)
    months = (np.arange(12) + 0.5) / 12.0
    return (years[:, None] + months[None, :]).ravel()
