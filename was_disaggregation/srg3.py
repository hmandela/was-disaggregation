"""Three-state semiparametric rainfall generator (Houngnibo et al., 2023).

The paper defines dry (<= 0.1 mm), wet, and extremely wet (>= the monthly
80th percentile) states, a first-order Markov chain reset each month, and a
Gaussian-kernel amount model with Silverman's bandwidth.  Historical years
receive the same forecast-conditioned weights as the other generators.

Implementation choices and sensitivity variants:

* By default the 80th percentile is computed among *wet* amounts in the
  conditioned historical record. ``extreme_quantile_basis='all'`` offers
  the article's literal daily-amount wording. Including dry zeros can make
  the extreme cutoff zero. Quantile tie/interpolation choices are explicit.
* Each state's KDE uses weighted historical daily amounts and an effective
  sample size in Silverman's rule.  This is the deterministic infinite-year-
  bootstrap analogue, not a literal random 1000-year resample. The alternative
  ``bandwidth_rule='resampled'`` uses the expected state-specific number of
  observations in that finite bootstrap, still without drawing its exact
  random record. The variance denominator is controlled by ``bandwidth_ddof``.
* Default KDE draws are conditioned on the state's amount interval. This
  avoids a simulated 'wet' amount being labelled 'extremely wet', or vice
  versa. ``state_interval='positive'`` and ``'raw'`` expose other conventions
  for sensitivity analysis; raw Gaussian KDE tails may include negative rain.
* Monthly initial probabilities are forecast-weighted historical fractions,
  following Equation (7).  Independently weighted transitions need not have
  these fractions as their exact stationary distribution; both are diagnosed.
  ``initial_mode='stationary'`` instead solves the invariant-distribution
  equations. Missing transition rows can be regularized or explicitly rejected.
* With ``state_from_amount=True``, the realized amount defines the next state;
  default latent-state propagation has a different probability law whenever
  the amount sampler permits crossing a state boundary.

These choices should be reported when comparing numerical results with the
published Kandi/Parakou experiments.  The model remains single-site in its
statistical assumptions; a correlated ``normal_draw`` adds a spatial extension.

Scientific attribution
----------------------
* Houngnibo, M. C. M., Ali, A., Agali, A., Waongo, M., Lawin, A. E., and
  Cohard, J.-M. (2023), "Stochastic disaggregation of seasonal precipitation
  forecasts of the West African Regional Climate Outlook Forum",
  International Journal of Climatology 43(12), 5569-5585,
  https://doi.org/10.1002/joc.8161, section 2.2, Equations (1), (5)-(7):
  three-state monthly chain, Gaussian KDE and forecast-conditioned record.
* Silverman, B. W. (1986), Density Estimation for Statistics and Data Analysis,
  Chapman and Hall: Gaussian normal-reference bandwidth
  h = 1.06 * standard_deviation * n**(-1/5), as used in Houngnibo et al.
  The weighted effective-size and interval-truncation variants here are
  package extensions; they are not a literal unbounded KDE bootstrap.

The printed expression for pi_1 in Houngnibo et al., Eq. (5), does not always
satisfy stationarity; diagnostics solve pi P = pi instead. The paper's
``paper_protocol`` configuration is expectation-based and includes documented
choices, so it is not a claim that every ambiguity in the original publication
or its pre-review scripts has a unique resolution. State-duration geometric
tails, historical support and seasonal-total calibration limits remain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.special import ndtr, ndtri


@dataclass(frozen=True)
class _KDE:
    """Components of a Gaussian KDE restricted to one precipitation state."""

    centres: np.ndarray
    cumulative: np.ndarray
    low_cdf: np.ndarray
    high_cdf: np.ndarray
    bandwidth: float
    lower: float
    upper: float


@dataclass
class SRG3Fit:
    """Monthly fit; site axis precedes the final state axes.

    ``transition`` is (month, site, previous_state, current_state), ``initial``
    is (month, site, state), ``kde`` is an object array (month, site, wet/extreme),
    and ``valid`` is (site,).  A fit selected for class-mixture conditioning
    gains a leading member axis on all arrays except ``months``.
    """

    months: np.ndarray
    transition: np.ndarray
    initial: np.ndarray
    extreme_threshold: np.ndarray
    kde: np.ndarray
    valid: np.ndarray
    wet_threshold: float = 0.1
    diagnostics: dict = field(default_factory=dict)
    reset_each_month: bool = True
    state_from_amount: bool = False

    def select(self, classes: np.ndarray, others: tuple["SRG3Fit", "SRG3Fit"]) -> "SRG3Fit":
        """Select below/normal/above class fits for each ``(member, site)``."""
        fits = (self,) + tuple(others)
        classes = np.asarray(classes)
        if (len(fits) != 3 or classes.ndim != 2 or
                classes.shape[1] != self.valid.shape[-1] or
                not np.isin(classes, (0, 1, 2)).all()):
            raise ValueError("classes must be (member, site) with values 0, 1, 2")
        if any(not np.array_equal(f.months, self.months) or
               f.wet_threshold != self.wet_threshold or f.valid.ndim != 1
               or f.reset_each_month != self.reset_each_month
               or f.state_from_amount != self.state_from_amount
               for f in fits):
            raise ValueError("class fits must have identical months, sites, and wet threshold")

        def pick(name):
            arrays = [getattr(f, name) for f in fits]
            a = arrays[0]
            selector = classes.reshape((classes.shape[0], 1, classes.shape[1]) +
                                       (1,) * (a.ndim - 2))
            return np.where(selector == 0, a[None],
                            np.where(selector == 1, arrays[1][None], arrays[2][None]))

        return SRG3Fit(
            months=self.months, transition=pick("transition"),
            initial=pick("initial"), extreme_threshold=pick("extreme_threshold"),
            kde=pick("kde"),
            valid=np.where(classes == 0, fits[0].valid[None],
                           np.where(classes == 1, fits[1].valid[None], fits[2].valid[None])),
            wet_threshold=self.wet_threshold,
            reset_each_month=self.reset_each_month,
            state_from_amount=self.state_from_amount,
            diagnostics={"mixture": "member-specific class parameters"},
        )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float,
                       method: str = "inverted_cdf") -> float:
    """Generalised inverse of the weighted empirical CDF."""
    order = np.argsort(values, kind="stable")
    sorted_values, sorted_weights = values[order], weights[order]
    if method == "hazen":
        positions = (np.cumsum(sorted_weights) - .5 * sorted_weights) / sorted_weights.sum()
        return float(np.interp(probability, positions, sorted_values))
    index = np.searchsorted(np.cumsum(sorted_weights), probability * sorted_weights.sum(), side="left")
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _stationary(transition: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Solve the invariant-distribution equations when the chain is identified."""
    matrix = transition.T - np.eye(3)
    matrix[-1] = 1.0
    rhs = np.array([0.0, 0.0, 1.0])
    try:
        result = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return fallback.copy()  # Reducible chains have no unique invariant law.
    if not np.isfinite(result).all() or np.min(result) < -1e-10:
        return fallback.copy()
    result = np.maximum(result, 0.0)
    return result / result.sum()


def _fit_kde(amounts: np.ndarray, weights: np.ndarray, lower: float, upper: float,
             *, bandwidth_n: float | None = None, bandwidth_ddof: int = 0) -> _KDE | None:
    if not amounts.size:
        return None
    weights = weights / weights.sum()
    effective_n = 1.0 / np.sum(weights * weights)
    mean = np.dot(weights, amounts)
    variance = np.dot(weights, (amounts - mean) ** 2)
    if bandwidth_ddof == 1 and effective_n > 1:
        variance /= 1 - np.sum(weights * weights)
    sigma = np.sqrt(variance)
    bandwidth_n = effective_n if bandwidth_n is None else bandwidth_n
    # The zero-bandwidth limiting distribution is a point mass. Artificial
    # jitter for identical observations invents variability absent in data.
    bandwidth = 1.06 * sigma * bandwidth_n ** (-0.2) if sigma > 1e-10 else 0.
    if bandwidth == 0:
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.
        return _KDE(amounts.copy(), cumulative, np.zeros_like(weights), np.ones_like(weights),
                    0., float(lower), float(upper))
    low_cdf = ndtr((lower - amounts) / bandwidth)
    high_cdf = (np.ones_like(low_cdf) if np.isinf(upper)
                else ndtr((upper - amounts) / bandwidth))
    component_mass = weights * np.maximum(high_cdf - low_cdf, 0.0)
    mass = component_mass.sum()
    if mass <= 0:
        # Can only occur from floating-point saturation at an unusually tight
        # interval; retaining an observed centre is preferable to a wrong state.
        closest = np.argmax(weights)
        component_mass = np.zeros_like(weights)
        component_mass[closest] = 1.0
    cumulative = np.cumsum(component_mass / component_mass.sum())
    cumulative[-1] = 1.0
    return _KDE(amounts.copy(), cumulative, low_cdf, high_cdf,
                float(bandwidth), float(lower), float(upper))


def fit_srg3(
    values: np.ndarray,
    month: np.ndarray,
    weights: np.ndarray,
    *,
    wet_threshold: float = 0.1,
    extreme_quantile: float = .8,
    extreme_quantile_basis: str = "wet",
    extreme_quantile_method: str = "inverted_cdf",
    bandwidth_rule: str = "effective",
    bandwidth_ddof: int = 0,
    resample_size: int = 1000,
    initial_mode: str = "fractions",
    state_interval: str = "truncate",
    missing_state_policy: str = "fallback",
    reset_each_month: bool = True,
    state_from_amount: bool = False,
) -> SRG3Fit:
    """Fit forecast-conditioned monthly three-state Markov and Gaussian KDEs.

    ``values`` is ``(year, day, site)`` and ``weights`` is ``(year, site)``.
    NaN observations are missing, not dry.  Every forecast-weighted month
    must contain an observed day for a site to be valid.  The conditional
    transition probability for each historical year is averaged with its
    forecast year weight (Equation 1), over years observing the given state.
    The extreme cutoff defaults to the weighted 80th percentile of wet days.
    ``extreme_quantile_basis='all'`` instead includes dry amounts, following
    the literal wording of the article; ties at/below the wet threshold may
    eliminate the intermediate wet state. ``extreme_quantile_method='hazen'``
    interpolates weighted mid-mass plotting positions. Default inverted-CDF
    quantiles retain the 0.9.0 convention.

    ``bandwidth_rule='effective'`` uses the historical effective sample size.
    ``'resampled'`` instead uses the expected number of state-specific amount
    observations in ``resample_size`` resampled years. These bandwidths are
    different: weighted likelihood fitting and KDE bandwidth selection are
    not both invariant to replication. ``bandwidth_ddof=1`` requests the
    usual reliability-weighted sample variance; 0 uses a population moment.
    ``initial_mode='stationary'`` solves pi P = pi rather than fitting annual
    state fractions. ``missing_state_policy='error'`` rejects unidentified
    rows; default fallback replaces them by the observed state fractions.

    ``state_interval='truncate'`` conditions each KDE on its state interval.
    ``'positive'`` only removes negative amounts; ``'raw'`` retains all
    Gaussian tails, including physically impossible negative rain. The last
    two are deliberate sensitivity experiments and may produce amounts whose
    state differs from the latent state. ``state_from_amount=True`` uses the
    realized state for the next transition, while default keeps the latent
    state. Diagnostics quantify the unbounded KDE mass outside its state.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    month = np.asarray(month)
    if values.ndim != 3 or not all(values.shape):
        raise ValueError("values must be nonempty (year, day, site)")
    ny, nd, ns = values.shape
    if weights.shape != (ny, ns):
        raise ValueError("weights must have shape (year, site)")
    if month.shape != (nd,) or not np.issubdtype(month.dtype, np.number) or not np.isfinite(month).all():
        raise ValueError("month must be a finite day vector")
    if np.any(month != month.astype(int)) or np.any((month < 1) | (month > 12)):
        raise ValueError("month must consist of integer calendar months 1..12")
    if not np.isfinite(wet_threshold) or wet_threshold <= 0:
        raise ValueError("wet_threshold must be finite and positive")
    if not np.isfinite(extreme_quantile) or not 0 < extreme_quantile < 1:
        raise ValueError("extreme_quantile must lie strictly between zero and one")
    if extreme_quantile_basis not in {"wet", "all"}:
        raise ValueError("extreme_quantile_basis must be 'wet' or 'all'")
    if extreme_quantile_method not in {"inverted_cdf", "hazen"}:
        raise ValueError("extreme_quantile_method must be 'inverted_cdf' or 'hazen'")
    if bandwidth_rule not in {"effective", "resampled"}:
        raise ValueError("bandwidth_rule must be 'effective' or 'resampled'")
    if bandwidth_ddof not in (0, 1) or isinstance(bandwidth_ddof, (bool, np.bool_)):
        raise ValueError("bandwidth_ddof must be 0 or 1")
    if isinstance(resample_size, (bool, np.bool_)) or not isinstance(resample_size, (int, np.integer)) \
            or resample_size < 1:
        raise ValueError("resample_size must be a positive integer")
    if initial_mode not in {"fractions", "stationary"}:
        raise ValueError("initial_mode must be 'fractions' or 'stationary'")
    if state_interval not in {"truncate", "positive", "raw"}:
        raise ValueError("state_interval must be 'truncate', 'positive' or 'raw'")
    if missing_state_policy not in {"fallback", "error"}:
        raise ValueError("missing_state_policy must be 'fallback' or 'error'")
    if any(not isinstance(v, (bool, np.bool_)) for v in (reset_each_month, state_from_amount)):
        raise ValueError("reset_each_month and state_from_amount must be bools")
    if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
        raise ValueError("rainfall must be nonnegative or NaN for missing")
    if np.isinf(weights).any() or np.any(weights[np.isfinite(weights)] < 0):
        raise ValueError("weights must be nonnegative or NaN for excluded years")
    weights = np.where(np.isfinite(weights), weights, 0.0)
    largest = weights.max(axis=0)
    weights = np.divide(weights, largest, out=np.zeros_like(weights), where=largest > 0)
    months = np.unique(month.astype(np.int16))
    nm = len(months)
    transition = np.full((nm, ns, 3, 3), np.nan)
    initial = np.full((nm, ns, 3), np.nan)
    extreme_threshold = np.full((nm, ns), np.nan)
    kde = np.empty((nm, ns, 2), dtype=object)
    kde.fill(None)
    available = np.zeros((nm, ns), bool)
    n_transition = np.zeros((nm, ns, 3), int)
    fallback_rows = np.zeros((nm, ns, 3), bool)
    stationary = np.full((nm, ns, 3), np.nan)
    bandwidths = np.full((nm, ns, 2), np.nan)
    unbounded_state_mismatch = np.full((nm, ns, 2), np.nan)
    for im, current_month in enumerate(months):
        selected = month == current_month
        x_month = values[:, selected, :]
        for site in range(ns):
            history = x_month[:, :, site]
            available_year = (weights[:, site] > 0) & np.isfinite(history).any(axis=1)
            if not available_year.any():
                continue
            available[im, site] = True
            observed = np.isfinite(history)
            wet = observed & (history > wet_threshold)
            yy, dd = np.where(wet & available_year[:, None])
            wet_amount = history[yy, dd]
            wet_weights = weights[yy, site]
            if wet_amount.size:
                if extreme_quantile_basis == "wet":
                    threshold_amount, threshold_weight = wet_amount, wet_weights
                else:
                    ty, td = np.where(observed & available_year[:, None])
                    threshold_amount, threshold_weight = history[ty, td], weights[ty, site]
                cutoff = _weighted_quantile(threshold_amount, threshold_weight, extreme_quantile,
                                            extreme_quantile_method)
                extreme_threshold[im, site] = cutoff
            else:
                cutoff = np.inf  # Entirely dry months require no amount model.
                extreme_threshold[im, site] = np.inf
            states = np.where(history <= wet_threshold, 0,
                              np.where(history < cutoff, 1, 2)).astype(np.int8)
            states[~observed] = -1
            counts = np.stack([(states == state).sum(axis=1) for state in range(3)], axis=1)
            days = observed.sum(axis=1)
            fractions = np.divide(counts, days[:, None],
                                  out=np.zeros_like(counts, dtype=float), where=days[:, None] > 0)
            active_weight = np.where(available_year, weights[:, site], 0.0)
            pi = active_weight @ fractions / active_weight.sum()
            initial[im, site] = pi / pi.sum()
            previous, following = states[:, :-1], states[:, 1:]
            valid_pair = (previous >= 0) & (following >= 0)
            # Selecting a month must not connect two nonadjacent dates through
            # an intervening month, even if that month occurs twice in input.
            valid_pair &= np.diff(np.flatnonzero(selected))[None, :] == 1
            for source in range(3):
                row_denom = np.sum(valid_pair & (previous == source), axis=1)
                identified = available_year & (row_denom > 0)
                n_transition[im, site, source] = row_denom[identified].sum()
                if identified.any():
                    row_count = np.stack([np.sum(valid_pair & (previous == source) &
                                                 (following == target), axis=1)
                                          for target in range(3)], axis=1)
                    yearly = row_count[identified] / row_denom[identified, None]
                    transition[im, site, source] = np.average(
                        yearly, axis=0, weights=weights[identified, site])
                else:
                    if missing_state_policy == "error":
                        raise ValueError(f"Unidentified SRG3 transition: month={current_month}, "
                                         f"site={site}, source_state={source}")
                    transition[im, site, source] = pi
                    fallback_rows[im, site, source] = True
            stationary[im, site] = _stationary(transition[im, site], initial[im, site])
            if initial_mode == "stationary":
                initial[im, site] = stationary[im, site]
            for state in (1, 2):
                keep = states[yy, dd] == state
                if not np.any(keep):
                    continue
                lower = (float(np.nextafter(wet_threshold, np.inf)) if state == 1
                         else max(float(cutoff), float(np.nextafter(wet_threshold, np.inf))))
                upper = float(cutoff) if state == 1 else np.inf
                physical_lower, physical_upper = lower, upper
                if state_interval == "positive":
                    lower, upper = 0., np.inf
                elif state_interval == "raw":
                    lower, upper = -np.inf, np.inf
                expected_n = (resample_size * np.dot(active_weight, counts[:, state])
                              / active_weight.sum()) if bandwidth_rule == "resampled" else None
                kde[im, site, state - 1] = _fit_kde(
                    wet_amount[keep], wet_weights[keep], lower, upper,
                    bandwidth_n=expected_n, bandwidth_ddof=bandwidth_ddof)
                model = kde[im, site, state - 1]
                bandwidths[im, site, state - 1] = model.bandwidth
                normalized = wet_weights[keep] / wet_weights[keep].sum()
                if model.bandwidth == 0:
                    interval_mass = ((model.centres >= physical_lower) &
                                     (model.centres < physical_upper)).astype(float)
                else:
                    interval_mass = (ndtr((physical_upper - model.centres) / model.bandwidth)
                                     - ndtr((physical_lower - model.centres) / model.bandwidth))
                unbounded_state_mismatch[im, site, state - 1] = 1 - np.dot(normalized, interval_mass)
    valid = available.all(axis=0)
    transition[:, ~valid] = np.nan
    initial[:, ~valid] = np.nan
    extreme_threshold[:, ~valid] = np.nan
    kde[:, ~valid] = None
    return SRG3Fit(
        months=months, transition=transition, initial=initial,
        extreme_threshold=extreme_threshold, kde=kde, valid=valid,
        wet_threshold=float(wet_threshold),
        reset_each_month=bool(reset_each_month), state_from_amount=bool(state_from_amount),
        diagnostics={"month_has_weighted_observations": available,
                     "observed_transition_count": n_transition,
                     "transition_row_fallback": fallback_rows,
                     "stationary_probability": stationary,
                     "initial_stationarity_gap": np.max(
                         np.abs(initial - stationary), axis=-1),
                     "extreme_threshold": extreme_threshold,
                     "extreme_quantile": float(extreme_quantile),
                     "extreme_quantile_basis": extreme_quantile_basis,
                     "extreme_quantile_method": extreme_quantile_method,
                     "bandwidth_rule": bandwidth_rule,
                     "bandwidth_ddof": int(bandwidth_ddof),
                     "resample_size": int(resample_size),
                     "kde_bandwidth": bandwidths,
                     "unbounded_kde_state_mismatch_probability": unbounded_state_mismatch,
                     "initial_mode": initial_mode,
                     "state_interval": state_interval,
                     "missing_state_policy": missing_state_policy},
    )


def _draw_kde(model: _KDE, u_component: np.ndarray, u_amount: np.ndarray) -> np.ndarray:
    index = np.searchsorted(model.cumulative, u_component, side="right")
    index = np.minimum(index, model.centres.size - 1)
    if model.bandwidth == 0:
        return model.centres[index].copy()
    p_low = model.low_cdf[index]
    p_high = model.high_cdf[index]
    quantile = np.clip(p_low + u_amount * (p_high - p_low),
                       np.finfo(float).eps, 1 - np.finfo(float).eps)
    result = model.centres[index] + model.bandwidth * ndtri(quantile)
    result = np.maximum(result, model.lower)
    if np.isfinite(model.upper):
        result = np.minimum(result, np.nextafter(model.upper, -np.inf))
    return result


def simulate_srg3(
    fit: SRG3Fit,
    month: np.ndarray,
    n_members: int,
    normal_draw: Callable[[int, int, int], np.ndarray],
) -> np.ndarray:
    """Generate rainfall ``(member, day, site)`` with monthly chain resets.

    ``normal_draw(n_members, day, stream)`` returns standard normals shaped
    ``(member, site)``.  Stream 0 selects a state, stream 1 selects a KDE
    component, and stream 1 at step ``day + n_days`` draws an independent
    truncated-Gaussian amount.  Reusing stream 1 with a disjoint time index
    keeps compatibility with the rainfall.py spatial-field callback.
    """
    month = np.asarray(month)
    if month.ndim != 1 or len(month) == 0 or np.any(~np.isin(month, fit.months)):
        raise ValueError("month must be a nonempty fitted calendar-month vector")
    if isinstance(n_members, (bool, np.bool_)) or not isinstance(n_members, (int, np.integer)) or n_members < 1:
        raise ValueError("n_members must be a positive integer")
    member_fit = fit.valid.ndim == 2
    if fit.valid.ndim not in (1, 2) or (member_fit and fit.valid.shape[0] != n_members):
        raise ValueError("fit.valid shape must be (site,) or (member, site)")
    ns = fit.valid.shape[-1]
    nd = month.size
    result = np.full((n_members, nd, ns), np.nan)
    previous = np.zeros((n_members, ns), dtype=np.int8)
    for day, current_month in enumerate(month):
        im = int(np.searchsorted(fit.months, current_month))
        monthly_reset = day == 0 or (fit.reset_each_month and current_month != month[day - 1])
        occurrence_normal = np.asarray(normal_draw(n_members, day, 0), dtype=float)
        u = np.clip(ndtr(occurrence_normal),
                    np.finfo(float).eps, 1 - np.finfo(float).eps)
        if u.shape != (n_members, ns):
            raise ValueError("normal_draw must return (n_members, n_sites)")
        pi = fit.initial[:, im] if member_fit else fit.initial[im][None]
        tr = fit.transition[:, im] if member_fit else np.broadcast_to(
            fit.transition[im], (n_members, ns, 3, 3))
        if monthly_reset:
            cdf = np.cumsum(pi, axis=-1)
        else:
            rows = np.take_along_axis(tr, previous[..., None, None], axis=-2)[..., 0, :]
            cdf = np.cumsum(rows, axis=-1)
        state = np.sum(u[..., None] > cdf, axis=-1).clip(0, 2).astype(np.int8)
        previous = state
        active = fit.valid if member_fit else np.broadcast_to(fit.valid, (n_members, ns))
        if not np.isfinite(occurrence_normal[active]).all():
            raise ValueError("normal_draw returned nonfinite values at valid sites")
        result[:, day][active & (state == 0)] = 0.0
        if not np.any(active & (state > 0)):
            continue
        component_normal = np.asarray(normal_draw(n_members, day, 1), dtype=float)
        amount_normal = np.asarray(normal_draw(n_members, day + nd, 1), dtype=float)
        uc = np.clip(ndtr(component_normal),
                     np.finfo(float).eps, 1 - np.finfo(float).eps)
        uq = np.clip(ndtr(amount_normal),
                     np.finfo(float).eps, 1 - np.finfo(float).eps)
        if uc.shape != (n_members, ns) or uq.shape != (n_members, ns):
            raise ValueError("normal_draw must return (n_members, n_sites)")
        if not np.isfinite(component_normal[active]).all() or not np.isfinite(amount_normal[active]).all():
            raise ValueError("normal_draw returned nonfinite amount values at valid sites")
        for site in range(ns):
            for target_state in (1, 2):
                chosen = np.flatnonzero(active[:, site] & (state[:, site] == target_state))
                if not chosen.size:
                    continue
                if member_fit:
                    groups = {}
                    for member in chosen:
                        model = fit.kde[member, im, site, target_state - 1]
                        if model is None:
                            raise ValueError("SRG3 sampled a state without an amount model")
                        key = id(model)
                        if key not in groups:
                            groups[key] = (model, [])
                        groups[key][1].append(member)
                    for model, members in groups.values():
                        members = np.asarray(members, dtype=int)
                        result[members, day, site] = _draw_kde(
                            model, uc[members, site], uq[members, site])
                else:
                    model = fit.kde[im, site, target_state - 1]
                    if model is None:
                        raise ValueError("SRG3 sampled a state without an amount model")
                    result[chosen, day, site] = _draw_kde(model, uc[chosen, site], uq[chosen, site])
        if fit.state_from_amount:
            cutoff = fit.extreme_threshold[:, im] if member_fit else fit.extreme_threshold[im][None]
            amounts = result[:, day]
            previous = np.where(amounts <= fit.wet_threshold, 0,
                                np.where(amounts < cutoff, 1, 2)).astype(np.int8)
    return result
