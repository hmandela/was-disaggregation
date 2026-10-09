"""Hindcast validation, proper scores and descriptive calibration diagnostics.

HindcastExperiment refits each method outside the target year, contiguous block,
or past-only verification window. Its orchestration and missing-data rules are
package infrastructure, not a reproduction of one author's hindcast experiment.
An in-sample run measures fit, not independent forecast skill.

Event thresholds and reference distributions are separate choices. Fixed
provider climatology defines the forecast event; cross_validated categories
estimate thresholds without the held-out year. reference_mode controls whether
the reference is fixed or training-only. Ties may give class frequencies other
than one third. Skill depends on the explicit reference and validation design.

References and implemented components
-------------------------------------
Edward S. Epstein (1969), "A Scoring System for Probability Forecasts of Ranked
Categories", Journal of Applied Meteorology 8, 985-987.
DOI: 10.1175/1520-0450(1969)008<0985:ASSFPF>2.0.CO;2.
    rps: unnormalized sum over the two nontrivial cumulative tercile events;
    score normalization conventions differ across applications.
Glenn W. Brier (1950), "Verification of Forecasts Expressed in Terms of
Probability", Monthly Weather Review 78, 1-3.
DOI: 10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2.
    brier_score: binary-event squared probability error.
Hans Hersbach (2000), "Decomposition of the Continuous Ranked Probability
Score for Ensemble Prediction Systems", Weather and Forecasting 15, 559-570.
DOI: 10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2.
    crps_ensemble: empirical ensemble CRPS; the paper's full decomposition
    is not implemented by this function.
Christopher A. T. Ferro (2014), "Fair scores for ensemble forecasts",
Quarterly Journal of the Royal Meteorological Society 140, 1917-1923.
DOI: 10.1002/qj.2270.
    fair CRPS, RPS and Brier estimators require conditionally iid members.
Allan H. Murphy (1973), "A New Vector Partition of the Probability Score",
Journal of Applied Meteorology 12, 595-600.
DOI: 10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2.
    brier_decomposition: reliability, resolution and uncertainty; a coarse
    binning residual is reported rather than silently claiming exact equality.
Thomas M. Hamill (2001), "Interpretation of Rank Histograms for Verifying
Ensemble Forecasts", Monthly Weather Review 129, 550-560.
DOI: 10.1175/1520-0493(2001)129<0550:IORHFV>2.0.CO;2.
    rank_histogram: randomized ties, or their fractional expectation. A flat
    histogram alone does not establish conditional calibration or joint skill.

These references identify scientific formulae; authorship of this software
remains distinct. Numerical tests do not reproduce the papers' experiments.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import ndtri

from .conditioning import classify_seasons
from .data import season_dates
from .dynamical import season_blocks
from .mre import classify_fixed

__all__ = ["HindcastExperiment", "training_years", "model_tercile_probabilities", "ensemble_normal_scores",
           "rps", "rps_ensemble", "crps_ensemble", "brier_score", "brier_decomposition",
           "longest_dry_spell", "rank_histogram", "reliability_table", "daily_scores"]


# ---------------------------------------------------------------------------
# small scoring and forecast-making utilities
# ---------------------------------------------------------------------------
def training_years(years, target, scheme="loyo", buffer=0, block_size=6):
    """Years a method may be fitted on when ``target`` is forecast.

    "loyo": all years but the target (and ``buffer`` neighbours on each side);
    "block": all years outside the contiguous block of ``block_size`` years that
    contains the target (k-fold cross-validation, for methods too expensive to
    refit every year, e.g. neural networks), again minus ``buffer`` years
    around the block; "past_only": only years strictly before the target,
    excluding its ``buffer`` immediately preceding years (expanding-window
    verification); "in_sample": all years."""
    years = [int(y) for y in years]
    if not years or len(set(years)) != len(years):
        raise ValueError("years must be nonempty and unique")
    if int(target) not in years:
        raise ValueError("target must be one of the supplied years")
    if not isinstance(buffer, (int, np.integer)) or buffer < 0:
        raise ValueError("buffer must be a nonnegative integer")
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if scheme == "in_sample":
        return years
    if scheme == "loyo":
        return [y for y in years if abs(y - int(target)) > int(buffer)]
    if scheme == "past_only":
        return [y for y in years if y < int(target) - int(buffer)]
    if scheme == "block":
        srt = sorted(years)
        i = srt.index(int(target)) // int(block_size)
        block = srt[i * int(block_size):(i + 1) * int(block_size)]
        lo, hi = min(block) - int(buffer), max(block) + int(buffer)
        return [y for y in years if not lo <= y <= hi]
    raise ValueError("scheme must be 'loyo', 'block', 'past_only' or 'in_sample'")


def model_tercile_probabilities(target, train, labels=("PB", "PN", "PA"), coords=None):
    """Tercile probabilities of a model ensemble in the model's own climate.

    target (member, site) totals of the forecast season; train (n, site) pooled
    model totals of the training years (all members). Returns (3, site), or a
    (probability, Y, X) DataArray when ``coords=(Y, X)`` is given."""
    target, train = np.asarray(target, float), np.asarray(train, float)
    if target.ndim != 2 or train.ndim != 2 or target.shape[1] != train.shape[1] or min(target.shape) == 0 or train.shape[0] == 0:
        raise ValueError("target and train must be nonempty (sample, site) arrays on the same sites")
    known = np.isfinite(train).any(0)
    thr = np.full((2, train.shape[1]), np.nan)
    if known.any():
        thr[:, known] = np.nanquantile(np.where(np.isfinite(train[:, known]), train[:, known], np.nan), [1 / 3, 2 / 3], axis=0)
    c = classify_fixed(target, thr)
    n = (c >= 0).sum(0)
    counts = np.stack([(c == k).sum(0) for k in range(3)]).astype(float)
    p = np.divide(counts, n, out=np.full_like(counts, np.nan), where=n > 0)
    if coords is None:
        return p
    yv, xv = coords
    return xr.DataArray(p.reshape(3, len(yv), len(xv)), dims=("probability", "Y", "X"),
                        coords={"probability": list(labels), "Y": yv, "X": xv})


def ensemble_normal_scores(target, train):
    """Normal scores (member, site) of target values within pooled training values (n, site)."""
    target, train = np.asarray(target, float), np.asarray(train, float)
    if target.ndim != 2 or train.ndim != 2 or target.shape[1] != train.shape[1]:
        raise ValueError("target and train must be (sample, site) arrays on the same sites")
    n = np.isfinite(train).sum(0)
    finite = np.isfinite(train)[None]
    below = (finite & (train[None] < target[:, None])).sum(1) + 0.5 * (finite & (train[None] == target[:, None])).sum(1)
    scores = ndtri((below + 1.0) / (n + 2.0))
    return np.where(np.isfinite(target) & (n > 0), scores, np.nan)


def rps(prob, obs_class):
    """Ranked probability score per site; prob (3, site), obs_class (site,) with -1 = missing.

    References
    ----------
    Edward S. Epstein (1969), DOI: 10.1175/1520-0450(1969)008<0985:ASSFPF>2.0.CO;2.
    Full titles and scientific scope appear in the module bibliography.
    """
    prob = np.asarray(prob, float)
    oc = np.asarray(obs_class)
    if prob.ndim != 2 or prob.shape[0] != 3 or oc.shape != (prob.shape[1],):
        raise ValueError("Need prob (3, site) and obs_class (site,)")
    onehot = np.stack([(oc == k) for k in range(3)]).astype(float)
    r = ((np.cumsum(prob, 0) - np.cumsum(onehot, 0)) ** 2)[:2].sum(0)
    valid = (np.isfinite(prob).all(0) & (prob >= 0).all(0)
             & np.isclose(prob.sum(0), 1.0, atol=1e-8))
    return np.where(np.isin(oc, [0, 1, 2]) & valid, r, np.nan)


def rps_ensemble(classes, obs_class, *, fair=False):
    """RPS from ``(member, site)`` integer tercile classes; -1/NaN are missing.

    ``fair=True`` subtracts the finite-iid-ensemble variance of both cumulative
    probabilities (Ferro 2014). At least two valid members are then required.
    Correlated or systematically resampled members do not meet the iid premise.

    References
    ----------
    Edward S. Epstein (1969); Christopher A. T. Ferro (2014), DOI: 10.1002/qj.2270.
    Full titles and scientific scope appear in the module bibliography.
    """
    c, o = np.asarray(classes, float), np.asarray(obs_class)
    if c.ndim != 2 or c.shape[0] < 1 or o.shape != (c.shape[1],):
        raise ValueError("Need nonempty classes (member, site) and obs_class (site,)")
    if (np.isfinite(c) & ~np.isin(c, [-1, 0, 1, 2])).any():
        raise ValueError("Classes must be 0,1,2, or -1/NaN for missing")
    valid = np.isin(c, [0, 1, 2])
    n = valid.sum(0)
    p = np.divide(np.stack([(c == k).sum(0) for k in range(3)]), n,
                  out=np.full((3, c.shape[1]), np.nan), where=n > 0)
    result = rps(p, o)
    if fair:
        cumulative = np.cumsum(p, axis=0)[:2]
        correction = np.divide((cumulative * (1 - cumulative)).sum(0), n - 1,
                               out=np.full(n.shape, np.nan), where=n > 1)
        result = result - correction
    return result


def crps_ensemble(ens, obs, *, fair=False):
    """CRPS of an ensemble (member, site); nonfinite members are ignored.

    The default scores the empirical forecast distribution (Hersbach 2000).
    ``fair=True`` estimates the CRPS of the underlying iid sampling law with
    denominator M(M-1), requiring M >= 2 (Ferro 2014). It must not be used to
    remove ensemble-size bias from correlated/systematically selected members.

    References
    ----------
    Hans Hersbach (2000), DOI: 10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2.
    Christopher A. T. Ferro (2014), DOI: 10.1002/qj.2270, for the fair option.
    Full titles and scientific scope appear in the module bibliography.
    """
    ens, obs = np.asarray(ens, float), np.asarray(obs, float)
    if ens.ndim != 2 or obs.shape != (ens.shape[1],) or ens.shape[0] == 0:
        raise ValueError("Need a nonempty ens (member, site) and obs (site,)")
    ens = np.where(np.isfinite(ens), ens, np.nan)
    ok = np.isfinite(ens)
    m = np.maximum(ok.sum(0), 1)
    t1 = np.nansum(np.abs(ens - obs[None]), 0) / m
    # The sorted-sample identity avoids a (member, member, site) allocation.
    ordered = np.sort(ens, axis=0)
    coefficients = 2 * np.arange(1, ens.shape[0] + 1)[:, None] - m[None] - 1
    denominator = m * (m - 1) if fair else m * m
    t2 = np.divide(np.nansum(coefficients * ordered, axis=0), denominator,
                   out=np.full(obs.shape, np.nan), where=denominator > 0)
    return np.where(np.isfinite(obs) & (ok.sum(0) >= (2 if fair else 1)), t1 - t2, np.nan)


def brier_score(prob, outcome, *, fair=False, ensemble_size=None):
    """Binary-event Brier score per case, optionally corrected for iid members.

    ``ensemble_size`` is the number of valid members used for each probability;
    fair correction is p(1-p)/(M-1). It applies only to event frequencies from
    iid members, not supplied/calibrated probabilities. Nonfinite pairs are NaN.

    References
    ----------
    Glenn W. Brier (1950), DOI: 10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2.
    Christopher A. T. Ferro (2014), DOI: 10.1002/qj.2270, for the fair option.
    Full titles and scientific scope appear in the module bibliography.
    """
    p, o = np.broadcast_arrays(np.asarray(prob, float), np.asarray(outcome, float))
    valid = np.isfinite(p) & np.isfinite(o)
    if (valid & ((p < 0) | (p > 1) | ~np.isin(o, [0, 1]))).any():
        raise ValueError("Finite probabilities must lie in [0,1] and outcomes must be binary")
    result = (p - o) ** 2
    if fair:
        if ensemble_size is None:
            raise ValueError("fair=True requires ensemble_size from iid members")
        n = np.broadcast_to(np.asarray(ensemble_size, float), p.shape)
        if (valid & (~np.isfinite(n) | (n < 2) | (n != np.floor(n)))).any():
            raise ValueError("Fair scores require an integer ensemble_size >= 2")
        result = result - p * (1 - p) / (n - 1)
    return np.where(valid, result, np.nan)


def brier_decomposition(prob, outcome, bins=None):
    """Murphy (1973) reliability - resolution + uncertainty decomposition.

    With ``bins=None``, cases are grouped by their exact probability and the
    identity is exact. With coarser bins, ``binning_residual`` makes the identity
    explicit; never label a coarse-bin three-term approximation as exact.
    Spatial/temporal pooling supplies a descriptive score, not independent
    samples for a significance test.

    References
    ----------
    Allan H. Murphy (1973), DOI: 10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2.
    Full titles and scientific scope appear in the module bibliography.
    """
    p, o = np.ravel(np.asarray(prob, float)), np.ravel(np.asarray(outcome, float))
    if p.shape != o.shape:
        raise ValueError("prob and outcome must have the same number of elements")
    score = brier_score(p, o)
    good = np.isfinite(score)
    p, o = p[good], o[good]
    if not p.size:
        return dict(score=np.nan, reliability=np.nan, resolution=np.nan,
                    uncertainty=np.nan, binning_residual=np.nan, count=0)
    if bins is None:
        _, group = np.unique(p, return_inverse=True)
    else:
        bins = np.asarray(bins, float)
        if bins.ndim != 1 or bins.size < 2 or not np.isfinite(bins).all() or (np.diff(bins) <= 0).any() or bins[0] > 0 or bins[-1] < 1:
            raise ValueError("bins must be strictly increasing and cover [0,1]")
        group = np.clip(np.digitize(p, bins) - 1, 0, bins.size - 2)
    counts = np.bincount(group).astype(float)
    used = counts > 0
    pbar = np.bincount(group, weights=p)[used] / counts[used]
    obar = np.bincount(group, weights=o)[used] / counts[used]
    weights = counts[used] / p.size
    climate = o.mean()
    rel = float(np.sum(weights * (pbar - obar) ** 2))
    res = float(np.sum(weights * (obar - climate) ** 2))
    unc = float(climate * (1 - climate))
    bs = float(np.mean((p - o) ** 2))
    return dict(score=bs, reliability=rel, resolution=res, uncertainty=unc,
                binning_residual=bs - (rel - res + unc), count=int(p.size))


def longest_dry_spell(x, wet_threshold=1.0):
    """Longest observed dry run along time (-2); missing values split runs.

    A partly observed run is a lower bound; wholly missing series stay NaN.
    """
    x = np.asarray(x, float)
    if x.ndim < 2 or not np.isfinite(wet_threshold) or wet_threshold <= 0:
        raise ValueError("Need (..., T, site) rainfall and a positive wet_threshold")
    run = np.zeros(x.shape[:-2] + x.shape[-1:])
    best = np.zeros_like(run)
    for t in range(x.shape[-2]):
        dry = np.isfinite(x[..., t, :]) & (x[..., t, :] < wet_threshold)
        run = np.where(dry, run + 1, 0)
        best = np.maximum(best, run)
    return np.where(np.isfinite(x).any(axis=-2), best, np.nan)


def _wet_frequency(x, axis, wet_threshold):
    valid = np.isfinite(x)
    count = valid.sum(axis=axis)
    wet = (valid & (x >= wet_threshold)).sum(axis=axis).astype(float)
    return np.divide(wet, count, out=np.full_like(wet, np.nan), where=count > 0)


def _finite_mean(x, axis=None):
    x = np.asarray(x, float)
    count = np.isfinite(x).sum(axis=axis)
    total = np.where(np.isfinite(x), x, 0.).sum(axis=axis)
    result = np.divide(total, count, out=np.full(np.shape(total), np.nan), where=count > 0)
    return float(result) if result.ndim == 0 else result


def rank_histogram(ens, obs, rng=None, *, ties="random"):
    """Rank histogram counts (M+1,) of observations (n,) within ensembles (M, n).

    Ties (e.g. zero rainfall in observation and members) are broken at random,
    as they must be for precipitation (Hamill 2001). ``ties="fractional"``
    distributes one case equally over its admissible ranks and returns the
    expectation of randomized tie breaking. Only complete ensemble cases enter
    the histogram; variable ensemble sizes require separate histograms.

    References
    ----------
    Thomas M. Hamill (2001), DOI: 10.1175/1520-0493(2001)129<0550:IORHFV>2.0.CO;2.
    Full titles and scientific scope appear in the module bibliography.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    ens, obs = np.asarray(ens, float), np.asarray(obs, float)
    if ens.ndim != 2 or ens.shape[0] < 1 or obs.shape != (ens.shape[1],):
        raise ValueError("Need nonempty ens (member, case) and obs (case,)")
    if ties not in {"random", "fractional"}:
        raise ValueError("ties must be 'random' or 'fractional'")
    ok = np.isfinite(obs) & np.isfinite(ens).all(0)
    e, o = ens[:, ok], obs[ok]
    below = (e < o[None]).sum(0)
    n_tied = (e == o[None]).sum(0)
    if ties == "fractional":
        counts = np.zeros(ens.shape[0] + 1)
        for low, nt in zip(below, n_tied):
            counts[low:low + nt + 1] += 1. / (nt + 1)
        return counts
    rank = below + np.floor(rng.random(o.size) * (n_tied + 1)).astype(int)
    return np.bincount(rank, minlength=ens.shape[0] + 1)


def reliability_table(prob, outcome, bins=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)):
    """Reliability diagram data: forecast probability bins, mean forecast probability,
    observed relative frequency and count. ``prob``/``outcome`` any shape (flattened)."""
    p, o = np.ravel(np.asarray(prob, float)), np.ravel(np.asarray(outcome, float))
    bins = np.asarray(bins, float)
    if p.shape != o.shape:
        raise ValueError("prob and outcome must have the same number of elements")
    if bins.ndim != 1 or len(bins) < 2 or not np.isfinite(bins).all() or (np.diff(bins) <= 0).any() or bins[0] > 0 or bins[-1] < 1:
        raise ValueError("bins must be strictly increasing and cover [0, 1]")
    ok = np.isfinite(p) & np.isfinite(o)
    p, o = p[ok], o[ok]
    if ((p < 0) | (p > 1) | (o < 0) | (o > 1)).any():
        raise ValueError("finite probabilities and outcomes must lie in [0, 1]")
    k = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
    rows = []
    for b in range(len(bins) - 1):
        w = k == b
        rows.append({"bin": f"[{bins[b]:.1f}, {min(bins[b + 1], 1):.1f})", "forecast": p[w].mean() if w.any() else np.nan,
                     "observed": o[w].mean() if w.any() else np.nan, "count": int(w.sum())})
    return pd.DataFrame(rows)


def daily_scores(obs, ens, wet_threshold=1.0, rng=None, *, include_fair=False):
    """Daily verification of an ensemble of fields.

    obs (T, site) and ens (M, T, site) in mm/day. Returns CRPS, MAE of the
    ensemble median, spread/error ratio, rank-histogram reliability index
    (sum |f_i - 1/(M+1)|, 0 = flat; Delle Monache et al. 2006) on wet
    observations and on all, wet-day frequency, mean wet-day intensity, the 99th
    percentile, the mean inter-site correlation, and the fraction of wet cells
    on days when the field mean is wet (spatial intermittency). Nonphysical
    rainfall is missing and descriptive comparisons use paired observed support.
    ``include_fair=True`` adds an iid-only CRPS estimate; do not request it for
    correlated or systematically selected members.
    """
    obs, ens = np.asarray(obs, float), np.asarray(ens, float)
    if obs.ndim != 2 or ens.ndim != 3 or ens.shape[1:] != obs.shape or ens.shape[0] == 0:
        raise ValueError("Need obs (T, site) and a nonempty ens (member, T, site)")
    if not np.isfinite(wet_threshold) or wet_threshold <= 0:
        raise ValueError("wet_threshold must be positive")
    obs = np.where(np.isfinite(obs) & (obs >= 0), obs, np.nan)
    ens = np.where(np.isfinite(ens) & (ens >= 0), ens, np.nan)
    paired = np.isfinite(obs) & np.isfinite(ens).any(0)
    obs = np.where(paired, obs, np.nan)
    ens = np.where(paired[None], ens, np.nan)
    M = ens.shape[0]
    ok = np.isfinite(obs)
    flat_e, flat_o = ens.reshape(M, -1), obs.ravel()
    crps = _finite_mean(crps_ensemble(flat_e, flat_o))
    mean = _finite_mean(ens, 0)
    rmse = np.sqrt(_finite_mean((mean - obs) ** 2))
    count = np.isfinite(ens).sum(0)
    variance = np.divide(np.nansum((ens - mean) ** 2, 0), count - 1,
                         out=np.full_like(mean, np.nan), where=count > 1)
    spread = np.sqrt(_finite_mean(variance)) if M > 1 else 0.0
    rh = rank_histogram(flat_e, flat_o, rng)
    f = rh / rh.sum() if rh.sum() else np.full(M + 1, np.nan)
    wet_o = flat_o >= wet_threshold
    rhw = rank_histogram(flat_e[:, wet_o], flat_o[wet_o], rng)
    fw = rhw / rhw.sum() if rhw.sum() else np.full(M + 1, np.nan)

    def site_corr(x):                                                           # (T, S) -> mean off-diagonal corr
        # Pairwise-complete observations avoid inventing dry days at missing sites.
        pairs = []
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                good = np.isfinite(x[:, i]) & np.isfinite(x[:, j])
                if good.sum() >= 2:
                    a, b = x[good, i], x[good, j]
                    if a.std() > 0 and b.std() > 0:
                        pairs.append(float(np.corrcoef(a, b)[0, 1]))
        return float(np.mean(pairs)) if pairs else np.nan

    def intermittency(x):
        n = np.isfinite(x).sum(1)
        field_mean = np.divide(np.nansum(x, 1), n, out=np.full(len(x), np.nan), where=n > 0)
        m = field_mean >= wet_threshold
        good = np.isfinite(x[m])
        return (x[m][good] >= wet_threshold).mean() if good.any() else np.nan

    median = np.full(flat_o.shape, np.nan)
    active = np.isfinite(flat_e).any(0)
    median[active] = np.nanmedian(flat_e[:, active], 0)
    wet_obs, wet_ens = obs[obs >= wet_threshold], ens[ens >= wet_threshold]
    finite_obs, finite_ens = obs[np.isfinite(obs)], ens[np.isfinite(ens)]
    result = {"CRPS": crps, "MAE (median)": _finite_mean(np.abs(median - flat_o)),
            "spread/error": spread / rmse if rmse > 0 else np.nan,
            "rank RI (all)": np.abs(f - 1 / (M + 1)).sum(), "rank RI (wet obs)": np.abs(fw - 1 / (M + 1)).sum(),
            "wet-day freq obs": float((obs[ok] >= wet_threshold).mean()) if ok.any() else np.nan,
            "wet-day freq sim": float((ens[np.isfinite(ens)] >= wet_threshold).mean()) if np.isfinite(ens).any() else np.nan,
            "intensity obs": float(wet_obs.mean()) if wet_obs.size else np.nan,
            "intensity sim": float(wet_ens.mean()) if wet_ens.size else np.nan,
            "q99 obs": float(np.quantile(finite_obs, 0.99)) if finite_obs.size else np.nan,
            "q99 sim": float(np.quantile(finite_ens, 0.99)) if finite_ens.size else np.nan,
            "site corr obs": site_corr(obs), "site corr sim": _finite_mean([site_corr(e) for e in ens]),
            "wet fraction obs": intermittency(obs), "wet fraction sim": _finite_mean([intermittency(e) for e in ens])}
    if include_fair:
        result["fair CRPS (iid)"] = _finite_mean(crps_ensemble(flat_e, flat_o, fair=True))
    return result


# ---------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------
_ACTIVE = {}


def _fold_worker(args):
    # forked workers: dask's threaded scheduler may hold locks copied at fork time
    name, year = args
    try:
        import dask
        with dask.config.set(scheduler="synchronous"):
            return _ACTIVE["exp"]._fold(name, year)
    except ImportError:                                                           # pragma: no cover
        return _ACTIVE["exp"]._fold(name, year)


class HindcastExperiment:
    """Leave-one-year-out (or in-sample) hindcast of several methods.

    observations : Dataset with PRCP, or DataArray (T, Y, X)
    years : hindcast years (year of the first month of the season)
    months : season months of every method's output (attributes must fit inside)
    attributes : {name: Attribute} to score, e.g. SeasonalTotal / OnsetDate / MaxDrySpell
    scheme : "loyo" (default), "block" (``block_size`` contiguous years left out
        together, k-fold), "past_only" (training precedes target) or "in_sample"
    buffer : also leave out ``buffer`` years on each side of the target
    reference_years : optional explicit season years used to define fixed
        tercile limits and climatological references. They are read from
        ``observations`` and may be outside ``years``. Pass e.g.
        ``range(1994, 2017)`` for a predeclared 1994–2016 baseline. If
        omitted, legacy fixed scoring uses all evaluated hindcast years,
        including the verified year. Overlap of explicit reference years
        with evaluated years is allowed for an established climatology;
        for strictly independent evaluation choose disjoint years.
    """

    def __init__(self, observations, years, months, attributes, scheme="loyo", buffer=0, wet_threshold=1.0,
                 block_size=6, reference_years=None, *, quantile_method="linear", tercile_method="empirical"):
        obs = observations["PRCP"] if isinstance(observations, xr.Dataset) else observations
        self.obs = obs.load()
        self.years = [int(y) for y in years]
        self.months = tuple(months)
        self.attributes = dict(attributes)
        self.scheme, self.buffer, self.thr = scheme, buffer, float(wet_threshold)
        self.block_size = block_size
        self.quantile_method, self.tercile_method = quantile_method, tercile_method
        if quantile_method not in {"linear", "hazen"} or tercile_method not in {"empirical", "gamma"}:
            raise ValueError("Use quantile_method linear/hazen and tercile_method empirical/gamma")
        if not self.years:
            raise ValueError("years must be nonempty")
        if not np.isfinite(self.thr) or self.thr <= 0:
            raise ValueError("wet_threshold must be positive")
        training_years(self.years, self.years[0], scheme, buffer, self.block_size)  # validates scheme
        if len(set(self.years)) != len(self.years):
            raise ValueError("years must be unique")
        if reference_years is None:
            self.reference_years_ = None
        else:
            self.reference_years_ = [int(y) for y in reference_years]
            if len(self.reference_years_) < 3 or len(set(self.reference_years_)) != len(self.reference_years_):
                raise ValueError("reference_years must contain at least three unique season years")
        self.dates_ = season_dates(self.years[0], self.months)
        self.yv_, self.xv_ = self.obs["Y"].values, self.obs["X"].values
        self.obs_attr_ = {a: self._attr_values(att, self.obs, self.years) for a, att in self.attributes.items()}
        self.fixed_reference_ = {}
        if self.reference_years_ is not None:
            for a, att in self.attributes.items():
                hist = self._attr_values(att, self.obs, self.reference_years_)
                complete = (np.isfinite(hist) & (hist >= 0)).sum(axis=0)
                if not np.any(complete >= 3):
                    raise ValueError(f"reference_years have fewer than three complete seasons for attribute {a!r}")
                _, thr = classify_seasons(hist, np.asarray(self.reference_years_),
                                           climatology=(min(self.reference_years_), max(self.reference_years_)),
                                           method=self.tercile_method, quantile_method=self.quantile_method)
                self.fixed_reference_[a] = (hist, thr)
        daily = season_blocks(self.obs, self.years, self.months)[:, 0]            # (year, T, site)
        self.obs_daily_ = {"longest dry spell": longest_dry_spell(daily, self.thr),
                           "wet-day frequency": _wet_frequency(daily, 1, self.thr)}
        self.methods, self.results_ = {}, {}

    @staticmethod
    def _attr_values(att, prcp, years):
        v = att.compute(prcp, years)
        dims = [d for d in v.dims if d not in ("Y", "X")]
        return v.transpose(*dims, "Y", "X").values.reshape(*[v.sizes[d] for d in dims], -1)

    def add(self, name, fit, predict=None):
        """Register a method: ``fit(train_years) -> model``, ``predict(model, year) -> ensemble``.
        Without ``predict``, ``fit(train_years, year)`` must return the ensemble."""
        self.methods[name] = (fit, predict)
        return self

    # --- running ---
    def _as_dataarray(self, ens, year):
        if isinstance(ens, xr.Dataset):
            ens = ens["PRCP"]
        if isinstance(ens, xr.DataArray):
            if "member" not in ens.dims:
                ens = ens.expand_dims(member=[0])
            if ens.sizes["member"] < 1 or set(ens.dims) != {"member", "T", "Y", "X"}:
                raise ValueError("Forecast must have nonempty member,T,Y,X dimensions")
            expected = {"T": np.asarray(season_dates(int(year), self.months)),
                        "Y": self.yv_, "X": self.xv_}
            for dim, wanted in expected.items():
                if not np.array_equal(ens[dim].values, wanted):
                    raise ValueError(f"Forecast {dim} coordinates must match the target season/grid exactly")
            return ens.transpose("member", "T", "Y", "X")
        a = np.asarray(ens, float)
        if a.ndim == 3:                                                           # (member, T, site)
            a = a.reshape(a.shape[0], a.shape[1], len(self.yv_), len(self.xv_))
        return xr.DataArray(a, dims=("member", "T", "Y", "X"),
                            coords={"member": np.arange(a.shape[0]), "T": season_dates(int(year), self.months),
                                    "Y": self.yv_, "X": self.xv_}, name="PRCP")

    def _fit(self, name, train):
        fit, predict = self.methods[name]
        key = (name, tuple(train))
        cache = getattr(self, "_cache", {})
        if key not in cache:
            cache.clear()                                                         # keep one fitted model
            cache[key] = fit(list(train))
            self._cache = cache
        return cache[key]

    def _fold(self, name, year):
        t0 = time.time()
        fit, predict = self.methods[name]
        train = training_years(self.years, year, self.scheme, self.buffer, self.block_size)
        if not train:
            raise ValueError("Cross-validation leaves no training years; reduce buffer/block_size or add years")
        ens = fit(list(train), year) if predict is None else predict(self._fit(name, train), year)
        da = self._as_dataarray(ens, year)
        out = {"attributes": {}}
        # attributes of an ensemble come back as (member, season_year, site) -> (member, site)
        for a, att in self.attributes.items():
            v = att.compute(da, [year])
            out["attributes"][a] = v.transpose("member", "season_year", "Y", "X").values.reshape(v.sizes["member"], -1)
        x = da.values.reshape(da.sizes["member"], da.sizes["T"], -1)
        out["daily"] = {"longest dry spell": longest_dry_spell(x, self.thr), "wet-day frequency": _wet_frequency(x, 1, self.thr)}
        out["train"], out["seconds"] = train, time.time() - t0
        return out

    def run(self, names=None, n_jobs=1, verbose=True):
        """Forecast every eligible year; ``past_only`` skips years without history."""
        names = list(self.methods) if names is None else list(names)
        for name in names:
            t0 = time.time()
            jobs = [(name, y) for y in self.years]
            if self.scheme == "past_only":
                jobs = [(n, y) for n, y in jobs
                        if training_years(self.years, y, self.scheme, self.buffer, self.block_size)]
                if not jobs:
                    raise ValueError("past_only leaves no verification years with earlier training data")
            if n_jobs > 1 and self.scheme == "loyo" and sys.platform.startswith("linux"):
                # Fork preserves locally defined fit/predict functions and large
                # loaded arrays without requiring them to be serializable.
                _ACTIVE["exp"] = self
                try:
                    with mp.get_context("fork").Pool(n_jobs) as pool:
                        res = pool.map(_fold_worker, jobs, chunksize=1)
                finally:
                    _ACTIVE.clear()
            else:
                if n_jobs > 1 and self.scheme == "loyo" and not sys.platform.startswith("linux"):
                    warnings.warn("n_jobs>1 uses sequential folds on Windows/macOS; "
                                  "fork is unavailable or unsafe there", RuntimeWarning, stacklevel=2)
                res = [self._fold(*j) for j in jobs]
            self.results_[name] = dict(zip((y for _, y in jobs), res))
            self._cache = {}
            if verbose:
                print(f"{name:32s} {self.scheme:9s} {len(jobs)} years  {time.time() - t0:6.1f}s")
        return self

    # --- scoring ---
    def _thresholds(self, a, train):
        idx = [self.years.index(y) for y in train]
        hist = self.obs_attr_[a][idx]
        _, thr = classify_seasons(hist, np.array(train), climatology=(min(train), max(train)),
                                  method=self.tercile_method, quantile_method=self.quantile_method)
        return hist, thr

    def _fixed_thresholds(self, a):
        """Fixed categories from an explicit baseline or legacy hindcast years."""
        return self.fixed_reference_[a] if self.reference_years_ is not None else self._thresholds(a, self.years)

    def _scoring_reference(self, a, train, year, categories, reference_mode):
        if reference_mode not in {"fixed", "leave_target_out", "past_only"}:
            raise ValueError("reference_mode must be 'fixed', 'leave_target_out' or 'past_only'")
        if categories == "cross_validated":
            hist, thr = self._thresholds(a, train)
            ref_years = np.asarray(train)
        else:
            hist, thr = self._fixed_thresholds(a)
            ref_years = np.asarray(self.reference_years_ if self.reference_years_ is not None else self.years)
        # Event thresholds are fixed independently of how the reference forecast
        # is constructed. This prevents changing the verified event per fold.
        if reference_mode == "leave_target_out":
            hist = hist[ref_years != year]
        elif reference_mode == "past_only":
            hist = hist[ref_years < year]
        return hist, thr

    def yearly(self, categories="fixed", *, reference_mode="fixed", fair=False):
        """Long table: method, year, attribute, RPS, RPS_clim, CRPS, CRPS_clim, obs, ens_mean (domain means).

        ``categories`` defines the verified events and the climatological reference:
        ``"fixed"`` (default): tercile limits and reference from all hindcast years,
        as operational categories are defined on a fixed climatology. The
        forecasts are still made without the target year; only the event
        definition includes it. The reference also includes the verified value;
        interpret skill relative to this finite-sample climatological benchmark.
        When ``reference_years`` was supplied, both limits and the reference
        instead come from that predeclared period, which may lie outside the
        verification period.
        ``"cross_validated"``: limits and reference from the training years only.
        This can make the left-out year more extreme relative to its reference,
        degrading the reference and inflating RPSS/CRPSS (the
        cross-validation degeneracy of Barnston & van den Dool 1993).
        ``reference_mode="leave_target_out"`` excludes the verified value from
        the climatological forecast while retaining fixed event thresholds;
        ``"past_only"`` also excludes future reference years. This does not make
        a modern, predefined climatology retrospectively available to an old
        forecast. ``fair=True`` corrects scores only under iid member sampling.
        """
        if categories not in {"fixed", "cross_validated"}:
            raise ValueError("categories must be 'fixed' or 'cross_validated'")
        if not self.results_:
            raise RuntimeError("Run hindcast methods before scoring")
        rows = []
        for name, res in self.results_.items():
            for y, r in res.items():
                i = self.years.index(y)
                for a in self.attributes:
                    hist, thr = self._scoring_reference(a, r["train"], y, categories, reference_mode)
                    o = self.obs_attr_[a][i]
                    oc = classify_fixed(o[None], thr)[0]
                    hc = classify_fixed(hist, thr)
                    e = r["attributes"][a]
                    ec = classify_fixed(e, thr)
                    prob = np.stack([(ec == k).sum(0) for k in range(3)]) / np.maximum((ec >= 0).sum(0), 1)
                    prob = np.where((ec >= 0).any(0)[None], prob, np.nan)
                    rs = rps_ensemble(ec, oc, fair=fair)
                    rc = rps_ensemble(hc, oc, fair=fair) if hist.shape[0] else np.full_like(o, np.nan)
                    cs = crps_ensemble(e, o, fair=fair)
                    cc = crps_ensemble(hist, o, fair=fair) if hist.shape[0] else np.full_like(o, np.nan)
                    rvalid, cvalid = np.isfinite(rs) & np.isfinite(rc), np.isfinite(cs) & np.isfinite(cc)
                    mean = _finite_mean(e, 0)
                    paired = np.isfinite(mean) & np.isfinite(o)
                    count = np.isfinite(e).sum(0)
                    variance = np.divide(np.nansum((e - mean) ** 2, 0), count - 1,
                                         out=np.full_like(mean, np.nan), where=count > 1)
                    spread = np.sqrt(variance)
                    rows.append(dict(method=name, year=y, attribute=a,
                                     RPS=_finite_mean(np.where(rvalid, rs, np.nan)), RPS_clim=_finite_mean(np.where(rvalid, rc, np.nan)),
                                     CRPS=_finite_mean(np.where(cvalid, cs, np.nan)), CRPS_clim=_finite_mean(np.where(cvalid, cc, np.nan)),
                                     obs=_finite_mean(np.where(paired, o, np.nan)), ens_mean=_finite_mean(np.where(paired, mean, np.nan)),
                                     spread=_finite_mean(np.where(paired, spread, np.nan)),
                                     spread2=_finite_mean(np.where(paired, variance, np.nan)),
                                     err2=_finite_mean((mean - o) ** 2),
                                     p_observed_class=_finite_mean(np.where(rvalid, prob[np.clip(oc, 0, 2), np.arange(oc.size)], np.nan))))
                for d in ("longest dry spell", "wet-day frequency"):
                    sim, ob = r["daily"][d], self.obs_daily_[d][i]
                    mean = _finite_mean(sim, 0)
                    paired = np.isfinite(mean) & np.isfinite(ob)
                    count = np.isfinite(sim).sum(0)
                    variance = np.divide(np.nansum((sim - mean) ** 2, 0), count - 1,
                                         out=np.full_like(mean, np.nan), where=count > 1)
                    spread = np.sqrt(variance)
                    rows.append(dict(method=name, year=y, attribute=d,
                                     obs=_finite_mean(np.where(paired, ob, np.nan)), ens_mean=_finite_mean(np.where(paired, mean, np.nan)),
                                     CRPS=_finite_mean(crps_ensemble(sim, ob, fair=fair)),
                                     spread=_finite_mean(np.where(paired, spread, np.nan)),
                                     spread2=_finite_mean(np.where(paired, variance, np.nan)),
                                     err2=_finite_mean((mean - ob) ** 2)))
        table = pd.DataFrame(rows)
        table.attrs.update(categories=categories, reference_mode=reference_mode,
                           score_estimator="fair iid" if fair else "empirical ensemble",
                           reference_overlap="fixed baselines can contain verified/future years")
        return table

    def reliability(self, attribute, category=2, categories="fixed", bins=(0, 0.2, 0.4, 0.6, 0.8, 1.0001)):
        """Reliability table of the forecast probability of tercile ``category``
        (0 low, 1 normal, 2 high) of ``attribute``, pooled over years and cells, per method."""
        if categories not in {"fixed", "cross_validated"} or category not in (0, 1, 2):
            raise ValueError("Use fixed/cross_validated categories and category 0, 1 or 2")
        out = {}
        for name, res in self.results_.items():
            P, O = [], []
            for y, r in res.items():
                hist, thr = (self._thresholds(attribute, r["train"]) if categories == "cross_validated"
                             else self._fixed_thresholds(attribute))
                oc = classify_fixed(self.obs_attr_[attribute][self.years.index(y)][None], thr)[0]
                ec = classify_fixed(r["attributes"][attribute], thr)
                n = (ec >= 0).sum(0)
                p = np.where(n > 0, (ec == category).sum(0) / np.maximum(n, 1), np.nan)
                P.append(p); O.append(np.where(oc >= 0, (oc == category).astype(float), np.nan))
            out[name] = reliability_table(np.concatenate(P), np.concatenate(O), bins)
        return pd.concat(out, names=["method", "row"])

    def rank_histograms(self, attribute, rng=None, *, ties="random"):
        """Rank histogram (relative frequencies) of the observed ``attribute`` within each
        method's ensemble, pooled over years and cells (ties broken at random)."""
        rng = np.random.default_rng(0) if rng is None else rng
        out = {}
        for name, res in self.results_.items():
            counts = None
            for y, r in res.items():
                if attribute in r["attributes"]:
                    e, o = r["attributes"][attribute], self.obs_attr_[attribute][self.years.index(y)]
                else:
                    e, o = r["daily"][attribute], self.obs_daily_[attribute][self.years.index(y)]
                h = rank_histogram(e, o, rng, ties=ties)
                if counts is not None and counts.shape != h.shape:
                    raise ValueError("Ensemble sizes differ between folds; use separate rank histograms for each size")
                counts = h if counts is None else counts + h
            out[name] = counts / counts.sum() if counts.sum() else np.full_like(counts, np.nan, dtype=float)
        return pd.DataFrame(out).T

    @staticmethod
    def _skill(g):
        out = {}
        # Each score has its own paired-valid years. Missing categorical
        # thresholds do not erase a valid continuous-score comparison.
        for score, reference, skill in (("RPS", "RPS_clim", "RPSS"), ("CRPS", "CRPS_clim", "CRPSS")):
            if score in g and reference in g:
                pair = g[[score, reference]].dropna()
                denominator = pair[reference].sum()
                out[skill] = 1 - pair[score].sum() / denominator if denominator > 0 else np.nan
        if "p_observed_class" in g:
            out["P(obs class)"] = g["p_observed_class"].mean()
        paired = g[["ens_mean", "obs"]].replace([np.inf, -np.inf], np.nan).dropna()
        out["corr"] = np.corrcoef(paired["ens_mean"], paired["obs"])[0, 1] if paired["obs"].std() > 0 and paired["ens_mean"].std() > 0 else np.nan
        out["bias"] = (g["ens_mean"] - g["obs"]).mean()
        err = np.sqrt(g["err2"].mean())
        spread = np.sqrt(g["spread2"].mean()) if "spread2" in g else g["spread"].mean()
        out["spread/error"] = spread / err if err > 0 else np.nan
        return pd.Series(out)

    def scores(self, n_boot=0, seed=0, categories="fixed", *, reference_mode="fixed", fair=False):
        """Scores per method and attribute, with equal weight per year.

        Scores first average over paired-valid cells in each year. RPSS / CRPSS
        are 1 - sum(score)/sum(reference) over all years; ``corr``
        is across years of domain means; ``spread/error`` = RMS ensemble sample
        SD / RMSE of the ensemble mean (per cell). With ``n_boot`` > 0, 5-95 %
        intervals of RPSS and CRPSS from resampling years are added.
        ``categories``: see ``yearly``."""
        if not isinstance(n_boot, (int, np.integer)) or n_boot < 0:
            raise ValueError("n_boot must be a nonnegative integer")
        yt = self.yearly(categories, reference_mode=reference_mode, fair=fair)
        # Construct an explicit row table: pandas GroupBy.apply can return a
        # DataFrame or a hierarchical Series depending on identical score keys.
        out = pd.DataFrame({key: self._skill(group)
                            for key, group in yt.groupby(["method", "attribute"], sort=False)}).T
        out.index = pd.MultiIndex.from_tuples(out.index, names=["method", "attribute"])
        if n_boot:
            rng = np.random.default_rng(seed)
            ys = np.asarray(sorted(yt["year"].unique()))
            boots = []
            for _ in range(n_boot):
                pick = rng.choice(ys, ys.size)
                g = pd.concat([yt[yt.year == y] for y in pick])
                estimates = {}
                for score, ref, skill in (("RPS", "RPS_clim", "RPSS"), ("CRPS", "CRPS_clim", "CRPSS")):
                    s = g.dropna(subset=[score, ref]).groupby(["method", "attribute"], sort=False)[[score, ref]].sum()
                    estimates[skill] = 1 - s[score] / s[ref].where(s[ref] > 0)
                boots.append(pd.DataFrame(estimates))
            b = pd.concat(boots, keys=range(n_boot), names=["boot"])
            q = b.groupby(level=["method", "attribute"]).quantile([0.05, 0.95]).unstack()
            q.columns = [f"{s} q{int(p * 100):02d}" for s, p in q.columns]
            out = out.join(q)
        return out
