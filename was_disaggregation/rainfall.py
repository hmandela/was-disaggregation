"""Monthly, forecast-conditioned first-order rainfall generator.

The forecast enters through historical year weights. Wet-day frequency is a
weighted mean of the monthly wet-day fractions of historical years. Gamma
parameters are estimated by weighted likelihood, the deterministic limiting
equivalent of fitting an infinitely large forecast-weighted year bootstrap.
Persistence is estimated from the unweighted historical record by default.

This implements the rainfall mechanism described in the supplied Part V text,
with explicit extensions: a Gamma law for *excess over* the wet threshold, an
atom for observations exactly at that threshold, and feasibility projection of
the occurrence transitions. It does not force seasonal totals to have exactly
the forecast tercile probabilities. Amount and occurrence latent fields are
supplied by the caller, so this module is independent of spatial field choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.special import digamma, gammaincinv, ndtr, polygamma


@dataclass
class RainFit:
    """Monthly parameters; parameter arrays have shape ``(month, site)``.

    ``valid`` is a one-dimensional site mask, or a (member, site) mask after
    selecting mixture classes. A site must have a positively
    weighted observed day in every fitted month. Dry sites are valid. Gamma
    ``shape`` and ``scale`` describe positive excess, in rainfall units;
    ``threshold_mass`` gives the point mass at exactly the wet threshold,
    conditional on a wet day. In dry months the amount parameters are unused.

    Diagnostic ``gamma_status`` codes are: 0 = weighted MLE; 1 = almost constant
    excess approximated with shape 1e6; 2 = no positive excess; 3 = exponential
    fallback after a numerical failure. Diagnostics are arrays by month/site.
    """

    months: np.ndarray
    pi: np.ndarray
    p01: np.ndarray
    p11: np.ndarray
    shape: np.ndarray
    scale: np.ndarray
    valid: np.ndarray
    threshold_mass: np.ndarray
    wet_threshold: float = 1.0
    diagnostics: dict = field(default_factory=dict)
    amount_distribution: str = "gamma"
    alpha: np.ndarray | None = None      # mixed exponential weight of component 1
    beta1: np.ndarray | None = None      # mixed exponential mean of component 1 (mm)
    beta2: np.ndarray | None = None      # mixed exponential mean of component 2 (mm)
    trace_probability: np.ndarray | None = None  # P(0 < PRCP < threshold | not wet)
    trace_mean: np.ndarray | None = None         # mean sub-threshold amount on trace days (mm)
    occurrence: str = "markov"                   # 'markov' (first order) or 'spell' (semi-Markov)
    dry_hazard: np.ndarray | None = None         # (month, max_dry_run, site): P(wet | dry run of L days)
    wet_continue: np.ndarray | None = None       # (month, max_wet_run, site): P(wet | wet run of L days)
    initial_wet: np.ndarray | None = None        # (site,) P(day before season start is wet)
    initial_dry_run: np.ndarray | None = None    # (site,) dry-run length before season start if dry

    def select(self, classes, others):
        """Member-specific parameters for mixture conditioning.

        ``self``/``others`` are the fits of classes (B, N, A) in that order;
        ``classes`` is (member, site) with values 0, 1, 2. Returns a RainFit
        whose (month, site) arrays become (member, month, site).
        """
        fits = (self,) + tuple(others)
        classes = np.asarray(classes)
        if (len(fits) != 3 or classes.ndim != 2 or classes.shape[1] != len(self.valid)
                or not np.isin(classes, [0, 1, 2]).all()):
            raise ValueError("classes must be (member,site) with values 0, 1, 2 and three class fits")
        def pick(name):
            arrays = [getattr(f, name) for f in fits]
            if arrays[0] is None:
                return None
            cls = classes.reshape((classes.shape[0],) + (1,) * (arrays[0].ndim - 1) + (classes.shape[1],))
            return np.where(cls == 0, arrays[0][None], np.where(cls == 1, arrays[1][None], arrays[2][None]))
        selected_valid = np.where(classes == 0, fits[0].valid[None],
                                  np.where(classes == 1, fits[1].valid[None], fits[2].valid[None]))
        return RainFit(months=self.months, pi=pick("pi"), p01=pick("p01"), p11=pick("p11"),
                       shape=pick("shape"), scale=pick("scale"),
                       valid=selected_valid,
                       threshold_mass=pick("threshold_mass"), wet_threshold=self.wet_threshold,
                       diagnostics={"mixture": "member-specific class parameters"},
                       amount_distribution=self.amount_distribution,
                       alpha=pick("alpha"), beta1=pick("beta1"), beta2=pick("beta2"),
                       trace_probability=pick("trace_probability"), trace_mean=pick("trace_mean"),
                       occurrence=self.occurrence, dry_hazard=pick("dry_hazard"), wet_continue=pick("wet_continue"),
                       initial_wet=pick("initial_wet"), initial_dry_run=pick("initial_dry_run"))


def _months(month: np.ndarray, n_days: int | None = None) -> np.ndarray:
    month = np.asarray(month)
    if month.ndim != 1 or month.size == 0:
        raise ValueError("month must be a nonempty one-dimensional day vector")
    if n_days is not None and month.size != n_days:
        raise ValueError("month length must equal the daily axis length")
    if not np.issubdtype(month.dtype, np.number):
        raise ValueError("month must contain integer calendar months")
    if not np.all(np.isfinite(month)) or np.any(month != month.astype(int)):
        raise ValueError("month must contain finite integer calendar months")
    if np.any((month < 1) | (month > 12)):
        raise ValueError("calendar months must lie between 1 and 12")
    return month.astype(np.int16)


def _gamma_mle(
    excess: np.ndarray, sample_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized, weighted, fixed-location Gamma MLE by sufficient statistics."""
    positive = np.isfinite(excess) & (excess > 0) & (sample_weights > 0)
    weighted = np.where(positive, sample_weights, 0.0)
    total = weighted.sum(axis=(0, 1))
    safe = np.where(positive, excess, 1.0)
    mean = np.divide(
        (weighted * safe).sum(axis=(0, 1)), total,
        out=np.ones(total.shape), where=total > 0,
    )
    mean_log = np.divide(
        (weighted * np.log(safe)).sum(axis=(0, 1)), total,
        out=np.zeros(total.shape), where=total > 0,
    )
    s = np.maximum(np.log(mean) - mean_log, 0.0)
    shape = np.ones(total.shape)
    scale = np.ones(total.shape)
    status = np.full(total.shape, 2, dtype=np.int8)
    active = (total > 0) & (s > 5.0e-7)
    constant = (total > 0) & ~active
    # The true fixed-location MLE diverges for identical positive observations.
    # A bounded concentration preserves the mean and limits numerical hazards.
    shape[constant] = 1.0e6
    scale[constant] = mean[constant] / shape[constant]
    status[constant] = 1
    if np.any(active):
        sa = s[active]
        # Positive starting estimate, then Newton solve log(k)-psi(k)=s.
        k = (3.0 - sa + np.sqrt((sa - 3.0) ** 2 + 24.0 * sa)) / (12.0 * sa)
        for _ in range(50):
            residual = np.log(k) - digamma(k) - sa
            derivative = 1.0 / k - polygamma(1, k)
            proposal = k - residual / derivative
            proposal = np.where((proposal > 0) & np.isfinite(proposal), proposal, k / 2)
            change = np.max(np.abs(proposal - k) / np.maximum(k, 1e-12))
            k = proposal
            if change < 1e-10:
                break
        converged = np.isfinite(k) & (k > 0)
        converged &= np.abs(np.log(k) - digamma(k) - sa) < 1e-7 * (1 + sa)
        k = np.where(converged, k, 1.0)
        shape[active] = k
        scale[active] = mean[active] / k
        status[active] = np.where(converged, 0, 3)
    # Weighted effective wet-amount sample count is descriptive, not a DOF
    # correction to the MLE. Equal year rescaling leaves all outputs invariant.
    sumsq = (weighted * weighted).sum(axis=(0, 1))
    effective_n = np.divide(total**2, sumsq, out=np.zeros_like(total), where=sumsq > 0)
    return shape, scale, status, effective_n


def _mixed_exponential_em(excess, sample_weights, alpha_fixed=None, init=None,
                          n_iter=500, tol=1e-8):
    """Weighted EM for f(x) = a/b1 exp(-x/b1) + (1-a)/b2 exp(-x/b2), x > 0.

    Vectorized over sites; ``excess``/``sample_weights`` are (year, day, site).
    With ``alpha_fixed`` (site,) only the two means are re-estimated, as Wilks
    (2002) did for forecast-conditioned fits. Returns alpha, beta1, beta2, status
    with status 0 = EM, 2 = no positive excess, 4 = single exponential fallback
    (fewer than 10 positive values or a collapsed component).
    """
    positive = np.isfinite(excess) & (excess > 0) & (sample_weights > 0)
    n_sites = excess.shape[-1]
    x = np.where(positive, excess, 1.0).reshape(-1, n_sites)
    w = np.where(positive, sample_weights, 0.0).reshape(-1, n_sites)
    total = w.sum(axis=0)
    mean = np.divide((w * x).sum(axis=0), total, out=np.ones(n_sites), where=total > 0)
    if init is not None:
        a, b1, b2 = (np.array(v, dtype=float, copy=True) for v in init)
        bad = ~(np.isfinite(a) & np.isfinite(b1) & np.isfinite(b2))
        a[bad], b1[bad], b2[bad] = 0.5, 0.4 * mean[bad], 2.0 * mean[bad]
    else:
        a, b1, b2 = np.full(n_sites, 0.5), 0.4 * mean, 2.0 * mean
    if alpha_fixed is not None:
        a = np.where(np.isfinite(alpha_fixed), alpha_fixed, a)
    b1, b2 = np.maximum(b1, 1e-3), np.maximum(b2, 1e-3)
    for _ in range(n_iter):
        f1 = a * np.exp(-x / b1) / b1
        f2 = (1 - a) * np.exp(-x / b2) / b2
        r = f1 / np.maximum(f1 + f2, 1e-300)
        wr, wq = w * r, w * (1 - r)
        s1, s2 = wr.sum(axis=0), wq.sum(axis=0)
        nb1 = np.divide((wr * x).sum(axis=0), s1, out=b1.copy(), where=s1 > 0)
        nb2 = np.divide((wq * x).sum(axis=0), s2, out=b2.copy(), where=s2 > 0)
        na = a if alpha_fixed is not None else np.clip(np.divide(s1, total, out=a.copy(), where=total > 0), 1e-4, 1 - 1e-4)
        change = np.max(np.abs(nb1 - b1) / np.maximum(b1, 1e-6) + np.abs(nb2 - b2) / np.maximum(b2, 1e-6) + np.abs(na - a))
        a, b1, b2 = na, np.maximum(nb1, 1e-3), np.maximum(nb2, 1e-3)
        if change < tol:
            break
    if alpha_fixed is None:                      # convention: beta1 <= beta2
        swap = b1 > b2
        a, b1, b2 = np.where(swap, 1 - a, a), np.where(swap, b2, b1), np.where(swap, b1, b2)
    count = positive.reshape(-1, n_sites).sum(axis=0)
    status = np.zeros(n_sites, dtype=np.int8)
    single = (count < 10) & (count > 0)
    a[single], b1[single], b2[single], status[single] = 1.0, mean[single], mean[single], 4
    none = count == 0
    a[none], b1[none], b2[none], status[none] = 1.0, 1.0, 1.0, 2
    return a, b1, b2, status


def mixed_exponential_ppf(u, alpha, beta1, beta2, n_iter=64):
    """Quantile of the mixed exponential by vectorized bisection."""
    target = 1.0 - u
    lo = np.zeros(np.broadcast(u, alpha, beta1, beta2).shape)
    hi = -np.maximum(beta1, beta2) * np.log(np.clip(target, 1e-300, None)) + lo
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        survival = alpha * np.exp(-mid / beta1) + (1 - alpha) * np.exp(-mid / beta2)
        above = survival > target
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    return 0.5 * (lo + hi)


def _run_lengths(wet, observed, initial_state):
    """Length of the run in progress at the END of day t-1, for every day t.

    Returns (prev_wet, prev_len, known) with shape (year, day, site); a run is
    'known' only if it started after the last missing value (or its start is
    given by ``initial_state``).
    """
    ny, nd, ns = wet.shape
    prev_wet = np.zeros(wet.shape, dtype=bool)
    prev_len = np.zeros(wet.shape, dtype=np.int32)
    known = np.zeros(wet.shape, dtype=bool)
    if initial_state is not None:
        w0, r0 = (np.asarray(v, float) for v in initial_state)
        cur_wet = np.nan_to_num(w0) > 0.5
        cur_len = np.where(np.isfinite(r0) & np.isfinite(w0), np.maximum(r0, 1), 0).astype(np.int32)
        cur_known = np.isfinite(w0) & np.isfinite(r0)
    else:
        cur_wet = np.zeros((ny, ns), dtype=bool)
        cur_len = np.zeros((ny, ns), dtype=np.int32)
        cur_known = np.zeros((ny, ns), dtype=bool)
    for d in range(nd):
        prev_wet[:, d], prev_len[:, d], known[:, d] = cur_wet, cur_len, cur_known
        obs_d, wet_d = observed[:, d], wet[:, d]
        continues = obs_d & (cur_len > 0) & (wet_d == cur_wet)
        # A new run's start is known only if the previous day was observed.
        new_known = cur_len > 0
        cur_known = np.where(continues, cur_known, obs_d & new_known)
        cur_len = np.where(continues, cur_len + 1, np.where(obs_d, 1, 0)).astype(np.int32)
        cur_wet = np.where(obs_d, wet_d, cur_wet)
    return prev_wet, prev_len, known


def _fit_spells(values, month, months, weights, wet_threshold, initial_state, max_dry, max_wet, prior):
    """Weighted run-length-dependent transition probabilities (semi-Markov)."""
    ny, nd, ns = values.shape
    observed = np.isfinite(values)
    wet = observed & (values >= wet_threshold)
    prev_wet, prev_len, known = _run_lengths(wet, observed, initial_state)
    # year-equivalent weights: uniform weights -> 1 per valid year
    w = np.asarray(weights, float)
    total = w.sum(axis=0, keepdims=True)
    w = np.divide(w * ny, total, out=np.zeros_like(w), where=total > 0)
    mon_index = np.searchsorted(months, month)
    risk = observed & known & (prev_len > 0)
    out = []
    for state, cap in ((False, max_dry), (True, max_wet)):
        at_risk = risk & (prev_wet == state)
        L = np.minimum(prev_len, cap) - 1
        idx = ((mon_index[None, :, None] * cap + L) * ns + np.arange(ns)[None, None, :])
        ww = np.broadcast_to(w[:, None, :], values.shape)
        size = len(months) * cap * ns
        n_risk = np.bincount(idx[at_risk], weights=ww[at_risk], minlength=size).reshape(len(months), cap, ns)
        n_wet = np.bincount(idx[at_risk & wet], weights=ww[at_risk & wet], minlength=size).reshape(len(months), cap, ns)
        # First-order transitions can use an observed predecessor even when
        # the start of its run is unknown (left-censored season or data gap).
        # Requiring known run starts for this fallback invents 50% wet chances
        # for fully dry/wet records.
        base_risk = np.zeros((len(months), 1, ns))
        base_wet = np.zeros_like(base_risk)
        pooled = np.zeros_like(base_risk)
        for im, mon in enumerate(months):
            days = month == mon
            raw_risk = observed[:, days] & (prev_len[:, days] > 0) & (prev_wet[:, days] == state)
            base_risk[im, 0] = np.where(raw_risk, ww[:, days], 0.).sum(axis=(0, 1))
            base_wet[im, 0] = np.where(raw_risk & wet[:, days], ww[:, days], 0.).sum(axis=(0, 1))
            obs_weight = np.where(observed[:, days], ww[:, days], 0.)
            den = obs_weight.sum(axis=(0, 1))
            pooled[im, 0] = np.divide((obs_weight * wet[:, days]).sum(axis=(0, 1)), den,
                                      out=np.full(ns, 0.5), where=den > 0)
        markov = np.divide(base_wet, base_risk, out=pooled, where=base_risk > 0)
        # shrink sparse run lengths towards the L-independent Markov estimate
        probability = np.divide(n_wet + prior * markov, n_risk + prior,
                                out=np.broadcast_to(markov, n_risk.shape).copy(),
                                where=n_risk + prior > 0)
        out.append(np.clip(probability, 0., 1.))
    first_known = known[:, 0] & np.isfinite(values[:, 0])
    w0 = np.where(first_known, w, 0.)
    s0 = w0.sum(axis=0)
    first_weight = np.where(observed[:, 0], w, 0.)
    first_total = first_weight.sum(0)
    initial_fallback = np.divide((first_weight * wet[:, 0]).sum(0), first_total,
                                 out=np.full(ns, 0.5), where=first_total > 0)
    initial_wet = np.divide((w0 * prev_wet[:, 0]).sum(0), s0, out=initial_fallback, where=s0 > 0)
    wd = w0 * ~prev_wet[:, 0]
    sd = wd.sum(0)
    initial_dry_run = np.divide((wd * np.minimum(prev_len[:, 0], max_dry)).sum(0), sd, out=np.ones(ns), where=sd > 0)
    return out[0], out[1], initial_wet, np.maximum(np.rint(initial_dry_run), 1)


def fit_rainfall(
    values: np.ndarray,
    month: np.ndarray,
    weights: np.ndarray,
    wet_threshold: float = 1.0,
    persistence: str = "climatology",
    amount_distribution: str = "gamma",
    alpha_fixed: np.ndarray | None = None,
    amount_init: tuple | None = None,
    include_trace: bool = False,
    occurrence: str = "markov",
    initial_state: tuple | None = None,
    max_dry_run: int = 40,
    max_wet_run: int = 15,
    spell_prior: float = 3.0,
) -> RainFit:
    """Fit rainfall using ``values[year, day, site]`` and ``weights[year, site]``.

    NaNs in rainfall are missing observations, never dry days. NaN year weights
    are excluded. Weights need not sum to one. Infinite or negative values and
    weights are rejected. A month with no weighted observations invalidates the
    site instead of inventing a climatology. An all-dry month remains valid.

    ``persistence='climatology'`` fits monthly transitions from all historical
    years, including years assigned zero forecast weight. Only adjacent days
    within the same historical year and calendar month enter those estimates.
    When one predecessor state is absent, persistence cannot be estimated and
    is set to zero. ``persistence='independent'`` explicitly sets it to zero.
    ``persistence='weighted'`` uses forecast-weighted transition counts, the
    empirical counterpart of Wilks's Eq. (3) (d as a function of the forecast).

    ``amount_distribution='mixed_exponential'`` fits Wilks's mixed exponential
    by weighted EM. ``alpha_fixed`` (month, site) holds the mixing weight at a
    climatological value (Wilks 2002); ``amount_init`` = (alpha, beta1, beta2)
    arrays (month, site) seed the EM.

    ``include_trace=True`` also fits sub-threshold ("trace") rainfall on
    non-wet days: its weighted frequency and mean. Without it every non-wet day
    is exactly zero, so simulated seasonal totals are biased low against
    terciles computed from the full observed totals (drizzle-rich reanalyses
    such as AgERA5 lose roughly 1-3 % of JAS rainfall this way).

    ``occurrence='spell'`` (0.4.0) replaces the first-order chain by a
    discrete-time semi-Markov (alternating renewal) chain: the probability that
    a dry spell ends, or a wet spell continues, depends on the month AND on the
    length L of the current run, h(L, month), estimated from forecast-weighted
    historical runs (each run weighted by its year's weight). This reproduces
    the weighted spell-length distributions (Racsko et al. 1991; Wilks 1999;
    LARS-WG uses the same idea with histograms), so dry-spell forecasts carried
    by the weights reach the simulated sequences. Runs >= max_dry_run /
    max_wet_run share the last bin. Sparse bins are shrunk towards the
    L-independent Markov estimate with ``spell_prior`` pseudo-days (in units of
    year-equivalent weights). ``initial_state`` = (wet, run) arrays
    (year, site) describing the day before the season (NaN if unknown), so that
    runs already in progress at season start are counted correctly.

    The climatological difference ``d=p11-p01`` is projected into the feasible
    interval at the conditioned wet fraction, then ``p01=pi*(1-d)`` and
    ``p11=pi+(1-pi)*d``. This preserves the stationary wet fraction exactly.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 3 or any(n == 0 for n in values.shape):
        raise ValueError("values must have nonempty dimensions (year, day, site)")
    if weights.shape != (values.shape[0], values.shape[2]):
        raise ValueError("weights must have dimensions (year, site)")
    month = _months(month, values.shape[1])
    if not np.isfinite(wet_threshold) or wet_threshold <= 0:
        raise ValueError("wet_threshold must be finite and strictly positive")
    if persistence not in {"climatology", "independent", "weighted"}:
        raise ValueError("persistence must be 'climatology', 'weighted' or 'independent'")
    if amount_distribution not in {"gamma", "mixed_exponential"}:
        raise ValueError("amount_distribution must be 'gamma' or 'mixed_exponential'")
    if occurrence not in {"markov", "spell"}:
        raise ValueError("occurrence must be 'markov' or 'spell'")
    if occurrence == "spell":
        for name, cap in (("max_dry_run", max_dry_run), ("max_wet_run", max_wet_run)):
            if isinstance(cap, (bool, np.bool_)) or not isinstance(cap, (int, np.integer)) or cap < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(spell_prior) or spell_prior < 0:
            raise ValueError("spell_prior must be finite and nonnegative")
        if initial_state is not None:
            if len(initial_state) != 2 or any(np.shape(v) != weights.shape for v in initial_state):
                raise ValueError("initial_state must be (wet, run) arrays with shape (year, site)")
            wet0, run0 = (np.asarray(v, dtype=float) for v in initial_state)
            if np.any(np.isinf(wet0)) or np.any(np.isinf(run0)):
                raise ValueError("initial_state must be finite or NaN for unknown history")
            if np.any(np.isfinite(wet0) & ~np.isin(wet0, [0, 1])):
                raise ValueError("initial_state wet must contain 0, 1, or NaN")
            if np.any(np.isfinite(run0) & ((run0 < 1) | (run0 != np.floor(run0)))):
                raise ValueError("initial_state run must contain positive integer run lengths or NaN")
    if np.any(np.isinf(values)) or np.any(values[np.isfinite(values)] < 0):
        raise ValueError("rainfall must be nonnegative and finite, or NaN for missing")
    if np.any(np.isinf(weights)) or np.any(weights[np.isfinite(weights)] < 0):
        raise ValueError("year weights must be nonnegative and finite, or NaN")
    weights = np.where(np.isfinite(weights), weights, 0.0)
    # Rescaling each site's weights avoids overflow and changes no fitted value.
    largest = weights.max(axis=0)
    weights = np.divide(weights, largest, out=np.zeros_like(weights), where=largest > 0)
    months = np.unique(month)
    n_sites = values.shape[2]
    parameter_shape = (len(months), n_sites)
    pi = np.full(parameter_shape, np.nan)
    d_raw = np.zeros(parameter_shape)
    d = np.zeros(parameter_shape)
    p01 = np.full(parameter_shape, np.nan)
    p11 = np.full(parameter_shape, np.nan)
    shape = np.ones(parameter_shape)
    scale = np.ones(parameter_shape)
    threshold_mass = np.ones(parameter_shape)
    available = np.zeros(parameter_shape, dtype=bool)
    gamma_status = np.full(parameter_shape, 2, dtype=np.int8)
    effective_n = np.zeros(parameter_shape)
    mix = amount_distribution == "mixed_exponential"
    trace_probability = np.zeros(parameter_shape) if include_trace else None
    trace_mean = np.zeros(parameter_shape) if include_trace else None
    alpha = np.full(parameter_shape, np.nan) if mix else None
    beta1 = np.full(parameter_shape, np.nan) if mix else None
    beta2 = np.full(parameter_shape, np.nan) if mix else None
    n_transition = np.zeros(parameter_shape, dtype=np.int64)
    persistence_identified = np.zeros(parameter_shape, dtype=bool)
    n_observed_years = np.zeros(parameter_shape, dtype=np.int64)
    for index, calendar_month in enumerate(months):
        selected = month == calendar_month
        x = values[:, selected, :]
        observed = np.isfinite(x)
        wet = observed & (x >= wet_threshold)
        n_valid = observed.sum(axis=1)
        year_fraction = np.divide(
            wet.sum(axis=1), n_valid, out=np.zeros(n_valid.shape, dtype=float), where=n_valid > 0,
        )
        w_year = weights * (n_valid > 0)
        w_total = w_year.sum(axis=0)
        available[index] = w_total > 0
        n_observed_years[index] = (w_year > 0).sum(axis=0)
        pi[index] = np.divide(
            (w_year * year_fraction).sum(axis=0), w_total,
            out=np.full(n_sites, np.nan), where=w_total > 0,
        )
        if persistence in {"climatology", "weighted"}:
            # Never connect one year's last day with another year's first day.
            pair_day = (month[1:] == calendar_month) & (month[:-1] == calendar_month)
            previous = values[:, :-1, :][:, pair_day, :]
            following = values[:, 1:, :][:, pair_day, :]
            pair_valid = np.isfinite(previous) & np.isfinite(following)
            previous_wet = previous >= wet_threshold
            following_wet = following >= wet_threshold
            pw = (np.broadcast_to(weights[:, None, :], pair_valid.shape) if persistence == "weighted"
                  else np.ones(pair_valid.shape))
            n0 = (pw * (pair_valid & ~previous_wet)).sum(axis=(0, 1))
            n1 = (pw * (pair_valid & previous_wet)).sum(axis=(0, 1))
            n01 = (pw * (pair_valid & ~previous_wet & following_wet)).sum(axis=(0, 1))
            n11 = (pw * (pair_valid & previous_wet & following_wet)).sum(axis=(0, 1))
            p01_hist = np.divide(n01, n0, out=np.zeros(n_sites), where=n0 > 0)
            p11_hist = np.divide(n11, n1, out=np.zeros(n_sites), where=n1 > 0)
            identified = (n0 > 0) & (n1 > 0)
            d_raw[index] = np.where(identified, p11_hist - p01_hist, 0.0)
            persistence_identified[index] = identified
            n_transition[index] = (pair_valid.sum(axis=(0, 1)) if persistence == "weighted"
                                   else np.rint(n0 + n1).astype(np.int64))
        p = np.nan_to_num(pi[index], nan=0.0)
        # Two transition constraints yield the lower bound for negative d.
        lo01 = np.divide(-(1 - p), p, out=np.full(n_sites, -np.inf), where=p > 0)
        lo11 = np.divide(-p, 1 - p, out=np.full(n_sites, -np.inf), where=p < 1)
        lower = np.maximum(lo01, lo11)
        d[index] = np.clip(d_raw[index], lower, 1.0)
        d[index] = np.where((p == 0) | (p == 1), 0.0, d[index])
        p01[index] = np.clip(p * (1 - d[index]), 0.0, 1.0)
        p11[index] = np.clip(p + (1 - p) * d[index], 0.0, 1.0)
        amount_weights = np.broadcast_to(weights[:, None, :], x.shape)
        if include_trace:
            not_wet = observed & ~wet
            trace = not_wet & (x > 0)
            w_dry = np.where(not_wet, amount_weights, 0.0).sum(axis=(0, 1))
            w_trace = np.where(trace, amount_weights, 0.0).sum(axis=(0, 1))
            trace_probability[index] = np.divide(w_trace, w_dry, out=np.zeros(n_sites), where=w_dry > 0)
            trace_mean[index] = np.divide(np.where(trace, amount_weights * x, 0.0).sum(axis=(0, 1)), w_trace,
                                          out=np.zeros(n_sites), where=w_trace > 0)
        wet_weight = np.where(wet, amount_weights, 0.0)
        wet_total = wet_weight.sum(axis=(0, 1))
        atom_weight = np.where(wet & (x == wet_threshold), amount_weights, 0.0).sum(axis=(0, 1))
        threshold_mass[index] = np.divide(
            atom_weight, wet_total, out=np.ones(n_sites), where=wet_total > 0,
        )
        shape[index], scale[index], gamma_status[index], effective_n[index] = _gamma_mle(
            x - wet_threshold, amount_weights,
        )
        if mix:
            fixed = None if alpha_fixed is None else np.asarray(alpha_fixed)[index]
            init = None if amount_init is None else tuple(np.asarray(v)[index] for v in amount_init)
            alpha[index], beta1[index], beta2[index], mix_status = _mixed_exponential_em(
                np.where(wet, x - wet_threshold, np.nan), amount_weights, alpha_fixed=fixed, init=init)
            gamma_status[index] = np.where(mix_status > 0, mix_status, gamma_status[index])
    dry_hazard = wet_continue = initial_wet = initial_dry_run = None
    if occurrence == "spell":
        dry_hazard, wet_continue, initial_wet, initial_dry_run = _fit_spells(
            values, month, months, weights, wet_threshold, initial_state, max_dry_run, max_wet_run, spell_prior)
    valid = available.all(axis=0)
    for parameter in (pi, p01, p11, shape, scale, threshold_mass, alpha, beta1, beta2,
                      trace_probability, trace_mean):
        if parameter is not None:
            parameter[:, ~valid] = np.nan
    return RainFit(
        months=months, pi=pi, p01=p01, p11=p11, shape=shape, scale=scale,
        valid=valid, threshold_mass=threshold_mass, wet_threshold=float(wet_threshold),
        amount_distribution=amount_distribution, alpha=alpha, beta1=beta1, beta2=beta2,
        trace_probability=trace_probability, trace_mean=trace_mean,
        occurrence=occurrence, dry_hazard=dry_hazard, wet_continue=wet_continue,
        initial_wet=initial_wet, initial_dry_run=initial_dry_run,
        diagnostics={
            "persistence_requested": persistence,
            "persistence_climatological": d_raw,
            "persistence_used": d,
            "persistence_projected": np.abs(d - d_raw) > 1e-12,
            "persistence_identified": persistence_identified,
            "valid_transition_count": n_transition,
            "weighted_observed_year_count": n_observed_years,
            "month_has_weighted_observations": available,
            "gamma_status": gamma_status,
            "positive_excess_effective_sample_size": effective_n,
        },
    )


def simulate_rainfall(
    fit: RainFit,
    month: np.ndarray,
    n_members: int,
    normal_draw: Callable[[int, int, int], np.ndarray],
    wet_threshold: float = 1.0,
) -> np.ndarray:
    """Simulate ``(member, day, site)`` using monthly Markov/Gamma parameters.

    ``normal_draw(n_members, step, stream)`` returns standard normal draws with
    shape ``(n_members, n_sites)``. Stream 0 controls occurrence, stream 1 amount.
    Spatial dependence comes entirely from that callback; draws must be fresh
    in time and occurrence/amount streams distinct. The first day uses the
    stationary wet fraction. Later days, including month boundaries, use the
    transition probabilities for the current month, so a changed monthly wet
    fraction induces a brief transition rather than resetting the chain.

    Uniform variates are bounded away from zero and one before Gamma inversion.
    Values below the wet threshold in the observations are modeled as dry zero.
    Parameter arrays may carry a leading member axis (member, month, site), as
    produced by :meth:`RainFit.select` for mixture conditioning.
    """
    month = _months(month)
    if isinstance(n_members, bool) or not isinstance(n_members, (int, np.integer)) or n_members < 1:
        raise ValueError("n_members must be a positive integer")
    if not np.isfinite(wet_threshold) or not np.isclose(wet_threshold, fit.wet_threshold, rtol=0, atol=1e-12):
        raise ValueError("simulation wet_threshold must match the fitted wet_threshold")
    n_sites = fit.valid.shape[-1]
    if fit.pi.ndim == 3 and fit.pi.shape[0] != n_members:
        raise ValueError("Member-specific fit parameters must match n_members")
    valid = np.broadcast_to(fit.valid, (n_members, n_sites))
    month_index = {int(value): index for index, value in enumerate(fit.months)}
    unknown = sorted(set(month.tolist()) - set(month_index))
    if unknown:
        raise ValueError(f"simulation requests months absent from fit: {unknown}")
    result = np.full((n_members, len(month), n_sites), np.nan)
    previous_wet = np.zeros((n_members, n_sites), dtype=bool)
    spell = fit.occurrence == "spell"
    if spell:
        cap_dry, cap_wet = fit.dry_hazard.shape[-2], fit.wet_continue.shape[-2]
        init_wet = np.asarray(fit.initial_wet, float)
        init_dry = np.asarray(fit.initial_dry_run, float)
        init_wet = init_wet if init_wet.ndim == 2 else init_wet[None, :]
        init_dry = init_dry if init_dry.ndim == 2 else init_dry[None, :]
        u0 = np.clip(ndtr(np.asarray(normal_draw(n_members, -3, 0), dtype=float)), 1e-15, 1 - 1e-15)
        previous_wet = u0 < np.nan_to_num(init_wet, nan=0.5)
        run = np.where(previous_wet, 1, np.nan_to_num(init_dry, nan=1.0)).astype(np.int32)
        run = np.broadcast_to(run, (n_members, n_sites)).copy()
        sites = np.arange(n_sites)[None, :]
        members = np.arange(n_members)[:, None]

        def hazard(table, index, length, cap):
            table = np.asarray(table)
            L = np.minimum(length, cap) - 1
            if table.ndim == 4:
                return table[members, index, L, sites]
            return table[index, L, sites]
    epsilon = np.finfo(float).eps
    def param(array, index):
        array = np.asarray(array)
        return array[:, index, :] if array.ndim == 3 else array[index][None, :]

    for step, calendar_month in enumerate(month):
        index = month_index[int(calendar_month)]
        uniforms = []
        for stream in (0, 1):
            normal = np.asarray(normal_draw(n_members, step, stream), dtype=float)
            if normal.shape != (n_members, n_sites):
                raise ValueError("normal_draw must return shape (n_members, n_sites)")
            if not np.all(np.isfinite(normal[valid])):
                raise ValueError("normal_draw returned nonfinite values at valid sites")
            uniforms.append(np.clip(ndtr(normal), epsilon, 1 - epsilon))
        u_occurrence, u_amount = uniforms
        if spell:
            occurrence_probability = np.where(previous_wet,
                                              hazard(fit.wet_continue, index, run, cap_wet),
                                              hazard(fit.dry_hazard, index, run, cap_dry))
        else:
            occurrence_probability = (
                param(fit.pi, index) if step == 0
                else np.where(previous_wet, param(fit.p11, index), param(fit.p01, index))
            )
        wet = (u_occurrence < occurrence_probability) & valid
        if spell:
            run = np.where(wet == previous_wet, run + 1, 1).astype(np.int32)
        atom = np.nan_to_num(param(fit.threshold_mass, index), nan=1.0)
        excess_probability = np.divide(
            u_amount - atom, 1 - atom,
            out=np.zeros_like(u_amount), where=(1 - atom) > 0,
        )
        excess_probability = np.clip(excess_probability, epsilon, 1 - epsilon)
        if fit.amount_distribution == "mixed_exponential":
            positive_amount = mixed_exponential_ppf(
                excess_probability, np.nan_to_num(param(fit.alpha, index), nan=1.0),
                np.nan_to_num(param(fit.beta1, index), nan=1.0), np.nan_to_num(param(fit.beta2, index), nan=1.0))
        else:
            shape = np.nan_to_num(param(fit.shape, index), nan=1.0)
            scale = np.nan_to_num(param(fit.scale, index), nan=1.0)
            positive_amount = scale * gammaincinv(shape, excess_probability)
        excess = np.where(u_amount <= atom, 0.0, positive_amount)
        dry_value = 0.0
        if fit.trace_probability is not None:
            # Non-wet days reuse the otherwise idle amount uniform. A bounded
            # uniform centered on the fitted trace mean preserves that mean
            # even when it exceeds half the wet threshold; clipping (0, 2*m)
            # at the threshold would bias trace rainfall downward.
            q = np.nan_to_num(param(fit.trace_probability, index))
            m = np.nan_to_num(param(fit.trace_mean, index))
            ratio = np.divide(u_amount, q, out=np.ones_like(u_amount), where=q > 0)
            half_width = np.minimum(m, wet_threshold - m)
            dry_value = np.where(u_amount < q, m + half_width * (2 * ratio - 1), 0.0)
            dry_value = np.minimum(dry_value, np.nextafter(wet_threshold, 0.))
        generated = np.where(wet, wet_threshold + excess, dry_value)
        generated = np.where(valid, generated, np.nan)
        result[:, step, :] = generated
        previous_wet = wet
    return result


def seasonal_total_moments(fit: RainFit, month: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate mean and variance of the simulated seasonal total per site.

    Each month is treated as a stationary two-state Markov chain with iid wet
    amounts (Katz 1985): Var(N_wet) ~ n pi (1-pi) (1+d)/(1-d), and
    Var(S) = n pi Var(A) + E[A]^2 Var(N_wet) without trace rainfall.
    When traces are enabled, dry-day mean/variance and the wet/dry mean
    contrast also contribute. Months are added as independent. For spell
    occurrence this Markov approximation does not use the fitted run hazards.
    Used to calibrate how much tercile-class variability a mixture may add.
    """
    month = _months(month)
    mean = np.zeros(len(fit.valid))
    var = np.zeros(len(fit.valid))
    for index, calendar_month in enumerate(fit.months):
        n = float((month == calendar_month).sum())
        pi = np.clip(np.nan_to_num(fit.pi[index]), 0, 1)
        d = np.clip(np.nan_to_num(np.asarray(fit.p11[index]) - np.asarray(fit.p01[index])), -0.95, 0.95)
        atom = np.clip(np.nan_to_num(fit.threshold_mass[index], nan=1.0), 0, 1)
        if fit.amount_distribution == "mixed_exponential":
            a, b1, b2 = (np.nan_to_num(np.asarray(v)[index], nan=1.0) for v in (fit.alpha, fit.beta1, fit.beta2))
            m1 = a * b1 + (1 - a) * b2
            m2 = 2 * (a * b1 ** 2 + (1 - a) * b2 ** 2)
        else:
            k, th = np.nan_to_num(fit.shape[index], nan=1.0), np.nan_to_num(fit.scale[index], nan=1.0)
            m1, m2 = k * th, k * (k + 1) * th ** 2
        ea = fit.wet_threshold + (1 - atom) * m1
        va = (1 - atom) * m2 - ((1 - atom) * m1) ** 2
        var_n = n * pi * (1 - pi) * (1 + d) / (1 - d)
        ed = vd = 0.
        if fit.trace_probability is not None:
            q = np.nan_to_num(fit.trace_probability[index])
            m = np.nan_to_num(fit.trace_mean[index])
            half_width = np.minimum(m, fit.wet_threshold - m)
            ed = q * m
            vd = q * (m ** 2 + half_width ** 2 / 3.) - ed ** 2
        mean += n * (pi * ea + (1 - pi) * ed)
        var += n * (pi * va + (1 - pi) * vd) + (ea - ed) ** 2 * var_n
    mean[~fit.valid], var[~fit.valid] = np.nan, np.nan
    return mean, var
