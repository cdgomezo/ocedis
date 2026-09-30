"""Background selection for the atmospheric Delta14C network.

Why this exists: the gridded Delta_atm is used as a TM5 forcing, and TM5
already simulates fossil dilution through CO2_ff x fossil_del = -1.0. A
Delta_atm carrying local fossil depletion would count that dilution twice in
the C_D14C budget, so the field must represent well-mixed background air.

Two filters, applied together:

  1. Site classification (ocean14c.stations.SITE_CLASS) -- a small hand-set
     list of sites with a known, dominant local fossil influence.
  2. Iterative one-sided outlier rejection, below.

The asymmetry in (2) is the whole point. Fossil CO2 is 14C-free, so
contamination can only push Delta14C *down*; genuine background variability is
roughly symmetric. Rejecting only the negative tail therefore removes
pollution events while leaving the background envelope intact. Rejecting
symmetrically would trim real high-side variability for no reason.
"""

import numpy as np

from ocean14c.curvefit import fit_station

DEFAULT_SIGMA = 3.0        # reject residuals below -k*sigma
DEFAULT_MAX_ITER = 5
DEFAULT_MIN_KEEP = 0.5     # refuse to reject more than half a record


def positive_side_scale(resid):
    """Robust residual scale estimated from the POSITIVE residuals only.

    Estimating sigma from all residuals and iterating is the classic runaway
    of sigma-clipping: each rejection shrinks sigma, which rejects more, and
    the loop never converges (it drove 8-16% rejection at remote sites like
    KUM and MLO, which is far too much for genuine pollution).

    Fossil contamination can only push Delta14C down, so the positive half of
    the residual distribution is uncontaminated. For a symmetric, zero-mean
    distribution, E[r^2 | r > 0] = sigma^2, so the RMS of the positive
    residuals estimates sigma directly -- and is immune to whatever the
    negative tail is doing.
    """
    resid = np.asarray(resid, dtype=float)
    pos = resid[resid > 0]
    if pos.size < 3:
        return float(np.std(resid))
    return float(np.sqrt(np.mean(pos**2)))


def filter_background(
    code,
    series,
    sigma=DEFAULT_SIGMA,
    max_iter=DEFAULT_MAX_ITER,
    min_keep=DEFAULT_MIN_KEEP,
    **fit_kwargs,
):
    """Iteratively reject negative outliers from one station's record.

    Fit with ccgcrv, compute residuals, drop points below -sigma*sd, refit,
    repeat until nothing more is rejected or `max_iter` is reached.

    Returns (fit, report). `fit` is the StationFit built from the retained
    points (None if the record is too short to fit at all). `report` is a dict
    with n_input / n_kept / n_rejected / reject_fraction / iterations /
    residual_sd -- a high reject_fraction is itself the signal that a site is
    not background.
    """
    report = {
        "code": code,
        "n_input": len(series),
        "n_kept": len(series),
        "n_rejected": 0,
        "reject_fraction": 0.0,
        "iterations": 0,
        "residual_sd": np.nan,
        "converged": False,
    }

    kept = list(series)
    rejected = []

    fit = fit_station(code, kept, **fit_kwargs)
    if fit is None:
        return None, report

    for iteration in range(1, max_iter + 1):
        t = np.array([d for d, _ in kept], dtype=float)
        v = np.array([x for _, x in kept], dtype=float)
        resid = v - fit.smooth(t)
        sd = positive_side_scale(resid)
        if not np.isfinite(sd) or sd == 0.0:
            report["converged"] = True
            break

        # One-sided: fossil contamination only depresses Delta14C.
        bad = resid < -sigma * sd
        if not bad.any():
            report["converged"] = True
            break

        # Never reject so much that the record stops being representative.
        if (np.count_nonzero(~bad) / len(series)) < min_keep:
            report["converged"] = True
            break

        rejected.extend([(t[i], v[i]) for i in np.flatnonzero(bad)])
        kept = [kept[i] for i in np.flatnonzero(~bad)]

        refit = fit_station(code, kept, **fit_kwargs)
        if refit is None:
            break
        fit = refit
        report["iterations"] = iteration

    if rejected:
        fit.rejected_t = np.array([t for t, _ in rejected])

    report["n_kept"] = len(kept)
    report["n_rejected"] = len(rejected)
    report["reject_fraction"] = len(rejected) / max(len(series), 1)
    report["residual_sd"] = fit.residual_sd
    return fit, report


def filter_all(stations, include_regional=False, **kwargs):
    """Background-filter every station in a registry.

    include_regional: if False (default), sites classified "regional" in
    ocean14c.stations.SITE_CLASS are dropped entirely rather than filtered.
    Set True to retain them -- ocean14c.krige will still downweight them via
    an inflated observation error.

    Returns (fits, reports) keyed by station code.
    """
    fits, reports = {}, {}
    for code, st in stations.items():
        if not include_regional and not st.is_background:
            continue
        fit, report = filter_background(code, st.series, **kwargs)
        report["site_class"] = st.site_class
        reports[code] = report
        if fit is not None:
            fits[code] = fit
    return fits, reports


def report_table(reports):
    """Human-readable background-filter summary."""
    lines = [
        f"{'code':6s} {'class':10s} {'n_in':>6s} {'n_kept':>7s} {'n_rej':>6s} "
        f"{'rej%':>6s} {'iter':>5s} {'sd':>6s}"
    ]
    for code in sorted(reports):
        r = reports[code]
        lines.append(
            f"{code:6s} {r.get('site_class','?'):10s} {r['n_input']:6d} "
            f"{r['n_kept']:7d} {r['n_rejected']:6d} "
            f"{100*r['reject_fraction']:6.1f} {r['iterations']:5d} "
            f"{r['residual_sd']:6.2f}"
        )
    return "\n".join(lines)
