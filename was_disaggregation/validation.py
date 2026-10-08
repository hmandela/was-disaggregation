"""Hindcast validation with leave-one-year-out refitting (0.7.0).

Every method that is fitted on the years it is then scored on is optimistic
in-sample: the bias corrector has seen the target season, the NHMM / NHSMM
has learnt its states and hazards from it, an analog or reweighting generator
can draw the target season itself, and a Schaake template can be the observed
season being forecast. ``HindcastExperiment`` refits each method without the
target year (optionally without ``buffer`` years on each side), forecasts
that year, and scores the ensemble against observations. By default the event
thresholds and reference use a fixed climatology; ``categories="cross_validated"``
instead estimates both from the training years only:

* RPS / RPSS of tercile classes of each season attribute (seasonal total,
  onset, dry spell, ...), reference = training-years class frequencies
  (not 1/3, attributes with ties have other climatological frequencies);
* CRPS / CRPSS of the attribute values, reference = the training-years
  observed values used as an ensemble;
* correlation of the ensemble mean with the observations across years
  (domain mean) and spread / error ratio;
* daily structure: longest dry spell and wet-day frequency, bias and
  interannual correlation.

A method is a pair ``fit(train_years) -> model`` and
``predict(model, year) -> ensemble`` (``(member, T, Y, X)`` DataArray or
Dataset with PRCP, or an array ``(member, T, site)``). With ``scheme="in_sample"``
the same pair is fitted once on all years, which measures the optimism.
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
           "rps", "crps_ensemble", "longest_dry_spell", "rank_histogram", "reliability_table", "daily_scores"]


# ---------------------------------------------------------------------------
# small scoring and forecast-making utilities
# ---------------------------------------------------------------------------
def training_years(years, target, scheme="loyo", buffer=0, block_size=6):
    """Years a method may be fitted on when ``target`` is forecast.

    "loyo": all years but the target (and ``buffer`` neighbours on each side);
    "block": all years outside the contiguous block of ``block_size`` years that
    contains the target (k-fold cross-validation, for methods too expensive to
    refit every year, e.g. neural networks), again minus ``buffer`` years
    around the block; "in_sample": all years."""
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
    if scheme == "block":
        srt = sorted(years)
        i = srt.index(int(target)) // int(block_size)
        block = srt[i * int(block_size):(i + 1) * int(block_size)]
        lo, hi = min(block) - int(buffer), max(block) + int(buffer)
        return [y for y in years if not lo <= y <= hi]
    raise ValueError("scheme must be 'loyo', 'block' or 'in_sample'")


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
    """Ranked probability score per site; prob (3, site), obs_class (site,) with -1 = missing."""
    prob = np.asarray(prob, float)
    oc = np.asarray(obs_class)
    onehot = np.stack([(oc == k) for k in range(3)]).astype(float)
    r = ((np.cumsum(prob, 0) - np.cumsum(onehot, 0)) ** 2)[:2].sum(0)
    valid = (np.isfinite(prob).all(0) & (prob >= 0).all(0)
             & np.isclose(prob.sum(0), 1.0, atol=1e-8))
    return np.where((oc >= 0) & (oc <= 2) & valid, r, np.nan)


def crps_ensemble(ens, obs):
    """CRPS of an ensemble (member, site) for observations (site,), NaN members ignored."""
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
    t2 = np.nansum(coefficients * ordered, axis=0) / (m * m)
    return np.where(np.isfinite(obs) & (ok.sum(0) > 0), t1 - t2, np.nan)


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
        dry = x[..., t, :] < wet_threshold
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


def rank_histogram(ens, obs, rng=None):
    """Rank histogram counts (M+1,) of observations (n,) within ensembles (M, n).

    Ties (e.g. zero rainfall in observation and members) are broken at random,
    as they must be for precipitation (Hamill 2001)."""
    rng = np.random.default_rng(0) if rng is None else rng
    ens, obs = np.asarray(ens, float), np.asarray(obs, float)
    ok = np.isfinite(obs) & np.isfinite(ens).all(0)
    e, o = ens[:, ok], obs[ok]
    below = (e < o[None]).sum(0)
    ties = (e == o[None]).sum(0)
    rank = below + np.floor(rng.random(o.size) * (ties + 1)).astype(int)
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


def daily_scores(obs, ens, wet_threshold=1.0, rng=None):
    """Daily verification of an ensemble of fields.

    obs (T, site) and ens (M, T, site) in mm/day. Returns CRPS, MAE of the
    ensemble median, spread/error ratio, rank-histogram reliability index
    (sum |f_i - 1/(M+1)|, 0 = flat; Delle Monache et al. 2006) on wet
    observations and on all, wet-day frequency, mean wet-day intensity, the 99th
    percentile, the mean inter-site correlation, and the fraction of wet cells
    on days when the field mean is wet (spatial intermittency)."""
    obs, ens = np.asarray(obs, float), np.asarray(ens, float)
    if obs.ndim != 2 or ens.ndim != 3 or ens.shape[1:] != obs.shape or ens.shape[0] == 0:
        raise ValueError("Need obs (T, site) and a nonempty ens (member, T, site)")
    if not np.isfinite(wet_threshold) or wet_threshold <= 0:
        raise ValueError("wet_threshold must be positive")
    obs = np.where(np.isfinite(obs), obs, np.nan)
    ens = np.where(np.isfinite(ens), ens, np.nan)
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
    return {"CRPS": crps, "MAE (median)": _finite_mean(np.abs(median - flat_o)),
            "spread/error": spread / rmse if rmse > 0 else np.nan,
            "rank RI (all)": np.abs(f - 1 / (M + 1)).sum(), "rank RI (wet obs)": np.abs(fw - 1 / (M + 1)).sum(),
            "wet-day freq obs": float((obs[ok] >= wet_threshold).mean()) if ok.any() else np.nan,
            "wet-day freq sim": float((ens[np.isfinite(ens)] >= wet_threshold).mean()) if np.isfinite(ens).any() else np.nan,
            "intensity obs": float(wet_obs.mean()) if wet_obs.size else np.nan,
            "intensity sim": float(wet_ens.mean()) if wet_ens.size else np.nan,
            "q99 obs": float(np.quantile(finite_obs, 0.99)) if finite_obs.size else np.nan,
            "q99 sim": float(np.quantile(finite_ens, 0.99)) if finite_ens.size else np.nan,
            "site corr obs": site_corr(obs), "site corr sim": np.mean([site_corr(e) for e in ens]),
            "wet fraction obs": intermittency(obs), "wet fraction sim": _finite_mean([intermittency(e) for e in ens])}


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
        together, k-fold) or "in_sample"
    buffer : also leave out ``buffer`` years on each side of the target
    """

    def __init__(self, observations, years, months, attributes, scheme="loyo", buffer=0, wet_threshold=1.0,
                 block_size=6):
        obs = observations["PRCP"] if isinstance(observations, xr.Dataset) else observations
        self.obs = obs.load()
        self.years = [int(y) for y in years]
        self.months = tuple(months)
        self.attributes = dict(attributes)
        self.scheme, self.buffer, self.thr = scheme, buffer, float(wet_threshold)
        self.block_size = block_size
        if not self.years:
            raise ValueError("years must be nonempty")
        if not np.isfinite(self.thr) or self.thr <= 0:
            raise ValueError("wet_threshold must be positive")
        training_years(self.years, self.years[0], scheme, buffer, self.block_size)  # validates scheme
        self.dates_ = season_dates(self.years[0], self.months)
        self.yv_, self.xv_ = self.obs["Y"].values, self.obs["X"].values
        self.obs_attr_ = {a: self._attr_values(att, self.obs, self.years) for a, att in self.attributes.items()}
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
        """Forecast every year with every (selected) method; results in ``results_``."""
        names = list(self.methods) if names is None else list(names)
        for name in names:
            t0 = time.time()
            jobs = [(name, y) for y in self.years]
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
            self.results_[name] = dict(zip(self.years, res))
            self._cache = {}
            if verbose:
                print(f"{name:32s} {self.scheme:9s} {len(self.years)} years  {time.time() - t0:6.1f}s")
        return self

    # --- scoring ---
    def _thresholds(self, a, train):
        idx = [self.years.index(y) for y in train]
        hist = self.obs_attr_[a][idx]
        _, thr = classify_seasons(hist, np.array(train), climatology=(min(train), max(train)))
        return hist, thr

    def yearly(self, categories="fixed"):
        """Long table: method, year, attribute, RPS, RPS_clim, CRPS, CRPS_clim, obs, ens_mean (domain means).

        ``categories`` defines the verified events and the climatological reference:
        ``"fixed"`` (default): tercile limits and reference from all hindcast years,
        as operational categories are defined on a fixed climatology. The
        forecasts are still made without the target year; only the event
        definition includes it. The reference also includes the verified value;
        interpret skill relative to this finite-sample climatological benchmark.
        ``"cross_validated"``: limits and reference from the training years only.
        This can make the left-out year more extreme relative to its reference,
        degrading the reference and inflating RPSS/CRPSS (the
        cross-validation degeneracy of Barnston & van den Dool 1993)."""
        if categories not in {"fixed", "cross_validated"}:
            raise ValueError("categories must be 'fixed' or 'cross_validated'")
        rows = []
        for name, res in self.results_.items():
            for y, r in res.items():
                i = self.years.index(y)
                for a in self.attributes:
                    hist, thr = self._thresholds(a, r["train"] if categories == "cross_validated" else self.years)
                    o = self.obs_attr_[a][i]
                    oc = classify_fixed(o[None], thr)[0]
                    hc = classify_fixed(hist, thr)
                    clim = np.stack([(hc == k).sum(0) for k in range(3)]) / np.maximum((hc >= 0).sum(0), 1)
                    e = r["attributes"][a]
                    ec = classify_fixed(e, thr)
                    prob = np.stack([(ec == k).sum(0) for k in range(3)]) / np.maximum((ec >= 0).sum(0), 1)
                    prob = np.where((ec >= 0).any(0)[None], prob, np.nan)
                    rs, rc = rps(prob, oc), rps(clim, oc)
                    cs, cc = crps_ensemble(e, o), crps_ensemble(hist, o)
                    rvalid, cvalid = np.isfinite(rs) & np.isfinite(rc), np.isfinite(cs) & np.isfinite(cc)
                    mean = _finite_mean(e, 0)
                    paired = np.isfinite(mean) & np.isfinite(o)
                    spread = np.sqrt(_finite_mean((e - mean) ** 2, 0))
                    rows.append(dict(method=name, year=y, attribute=a,
                                     RPS=_finite_mean(np.where(rvalid, rs, np.nan)), RPS_clim=_finite_mean(np.where(rvalid, rc, np.nan)),
                                     CRPS=_finite_mean(np.where(cvalid, cs, np.nan)), CRPS_clim=_finite_mean(np.where(cvalid, cc, np.nan)),
                                     obs=_finite_mean(np.where(paired, o, np.nan)), ens_mean=_finite_mean(np.where(paired, mean, np.nan)),
                                     spread=_finite_mean(np.where(paired, spread, np.nan)), err2=_finite_mean((mean - o) ** 2),
                                     p_observed_class=_finite_mean(np.where(rvalid, prob[np.clip(oc, 0, 2), np.arange(oc.size)], np.nan))))
                for d in ("longest dry spell", "wet-day frequency"):
                    sim, ob = r["daily"][d], self.obs_daily_[d][i]
                    mean = _finite_mean(sim, 0)
                    paired = np.isfinite(mean) & np.isfinite(ob)
                    spread = np.sqrt(_finite_mean((sim - mean) ** 2, 0))
                    rows.append(dict(method=name, year=y, attribute=d,
                                     obs=_finite_mean(np.where(paired, ob, np.nan)), ens_mean=_finite_mean(np.where(paired, mean, np.nan)),
                                     CRPS=_finite_mean(crps_ensemble(sim, ob)),
                                     spread=_finite_mean(np.where(paired, spread, np.nan)), err2=_finite_mean((mean - ob) ** 2)))
        return pd.DataFrame(rows)

    def reliability(self, attribute, category=2, categories="fixed", bins=(0, 0.2, 0.4, 0.6, 0.8, 1.0001)):
        """Reliability table of the forecast probability of tercile ``category``
        (0 low, 1 normal, 2 high) of ``attribute``, pooled over years and cells, per method."""
        if categories not in {"fixed", "cross_validated"} or category not in (0, 1, 2):
            raise ValueError("Use fixed/cross_validated categories and category 0, 1 or 2")
        out = {}
        for name, res in self.results_.items():
            P, O = [], []
            for y, r in res.items():
                hist, thr = self._thresholds(attribute, r["train"] if categories == "cross_validated" else self.years)
                oc = classify_fixed(self.obs_attr_[attribute][self.years.index(y)][None], thr)[0]
                ec = classify_fixed(r["attributes"][attribute], thr)
                n = (ec >= 0).sum(0)
                p = np.where(n > 0, (ec == category).sum(0) / np.maximum(n, 1), np.nan)
                P.append(p); O.append(np.where(oc >= 0, (oc == category).astype(float), np.nan))
            out[name] = reliability_table(np.concatenate(P), np.concatenate(O), bins)
        return pd.concat(out, names=["method", "row"])

    def rank_histograms(self, attribute, rng=None):
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
                h = rank_histogram(e, o, rng)
                counts = h if counts is None else counts + h
            out[name] = counts / counts.sum() if counts.sum() else np.full_like(counts, np.nan, dtype=float)
        return pd.DataFrame(out).T

    @staticmethod
    def _skill(g):
        out = {}
        if "RPS" in g and g["RPS"].notna().any():
            # Forecast and reference must be scored on the same valid years.
            for score, reference, skill in (("RPS", "RPS_clim", "RPSS"), ("CRPS", "CRPS_clim", "CRPSS")):
                pair = g[[score, reference]].dropna()
                denominator = pair[reference].sum()
                out[skill] = 1 - pair[score].sum() / denominator if denominator > 0 else np.nan
            out["P(obs class)"] = g["p_observed_class"].mean()
        paired = g[["ens_mean", "obs"]].replace([np.inf, -np.inf], np.nan).dropna()
        out["corr"] = np.corrcoef(paired["ens_mean"], paired["obs"])[0, 1] if paired["obs"].std() > 0 and paired["ens_mean"].std() > 0 else np.nan
        out["bias"] = (g["ens_mean"] - g["obs"]).mean()
        err = np.sqrt(g["err2"].mean())
        out["spread/error"] = g["spread"].mean() / err if err > 0 else np.nan
        return pd.Series(out)

    def scores(self, n_boot=0, seed=0, categories="fixed"):
        """Scores per method and attribute, with equal weight per year.

        Scores first average over paired-valid cells in each year. RPSS / CRPSS
        are 1 - sum(score)/sum(reference) over all years; ``corr``
        is across years of domain means; ``spread/error`` = mean ensemble SD /
        RMSE of the ensemble mean (per cell). With ``n_boot`` > 0, 5-95 %
        intervals of RPSS and CRPSS from resampling years are added.
        ``categories``: see ``yearly``."""
        yt = self.yearly(categories)
        out = yt.groupby(["method", "attribute"], sort=False).apply(self._skill).unstack()
        if n_boot:
            rng = np.random.default_rng(seed)
            ys = np.array(self.years)
            boots = []
            for _ in range(n_boot):
                pick = rng.choice(ys, ys.size)
                g = pd.concat([yt[yt.year == y] for y in pick])
                s = g.dropna(subset=["RPS"]).groupby(["method", "attribute"], sort=False)[["RPS", "RPS_clim", "CRPS", "CRPS_clim"]].sum()
                boots.append(pd.DataFrame({"RPSS": 1 - s.RPS / s.RPS_clim, "CRPSS": 1 - s.CRPS / s.CRPS_clim}))
            b = pd.concat(boots, keys=range(n_boot), names=["boot"])
            q = b.groupby(level=["method", "attribute"]).quantile([0.05, 0.95]).unstack()
            q.columns = [f"{s} q{int(p * 100):02d}" for s, p in q.columns]
            out = out.join(q)
        return out
