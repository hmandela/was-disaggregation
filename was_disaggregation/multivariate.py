"""Forecast-conditioned multivariate weather accompanying generated rainfall.

This is a Richardson-style extension of the supplied Wilks (2002) discussion.
Forecast year weights affect quadratic wet/dry means and log-variances; the local
VAR(1) and spatial dependence are fitted from unweighted historical residuals.
Humidity, wind and radiation are extensions, not an exact reproduction of that
paper. In particular, humidity uses a logit transform, and wind and solar use
log1p. Solar radiation has no astronomical upper bound in this implementation.

Array conventions are (year, day, site) for training and (member, day, site)
for simulation. Temperatures must already be in degrees Celsius, humidity in
percent, wind in m/s, and solar radiation in the caller's nonnegative unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import expit, logit


VARIABLE_ORDER = ("TMIN", "TMAX", "HUMIN", "HUMAX", "WIND", "SOLAR")


@dataclass
class MultiFit:
    """Fitted curves and a stable local residual process.

    ``mean`` and ``std`` have dimensions (variable, wet_state, day, site), where
    wet_state=0 is dry and 1 is wet. Matrix arrays have dimensions
    (month, site, variable, variable). ``valid`` has dimensions (variable, site),
    or (member, variable, site) after selecting mixture classes.
    Climatological arrays retain missing observations rather than imputing them.

    ``climatological_innovations`` contains whitened innovations in the exact
    variable order used by the normal-draw streams (10 + variable index). These
    are a preferable target for fitting variable-specific spatial kernels.
    """

    variables: tuple[str, ...]
    months: np.ndarray
    day_month: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    transition: np.ndarray
    innovation_cholesky: np.ndarray
    initial_cholesky: np.ndarray
    valid: np.ndarray
    climatological_residuals: dict[str, np.ndarray]
    climatological_innovations: dict[str, np.ndarray]
    diagnostics: dict = field(default_factory=dict)
    simulation_diagnostics: dict = field(default_factory=dict)

    @property
    def residuals(self) -> dict[str, np.ndarray]:
        """Alias for the unweighted, transformed standardized residuals."""
        return self.climatological_residuals

    @property
    def innovations(self) -> dict[str, np.ndarray]:
        """Whitened innovations suitable for spatial stream calibration."""
        return self.climatological_innovations

    def select(self, classes, others):
        """Member-specific curves for mixture conditioning.

        ``self``/``others`` are the fits of classes (B, N, A); they must share the
        climatological dependence (fit them with ``dependence=``). ``classes`` is
        (member, site). ``mean``/``std`` become (member, variable, state, day, site).
        """
        fits = (self,) + tuple(others)
        classes = np.asarray(classes)
        if (len(fits) != 3 or classes.ndim != 2 or classes.shape[1] != self.valid.shape[1]
                or not np.isin(classes, [0, 1, 2]).all()):
            raise ValueError("classes must be (member,site) with values 0, 1, 2 and three class fits")
        cls = classes[:, None, None, None, :]
        def pick(name):
            a = [getattr(f, name) for f in fits]
            return np.where(cls == 0, a[0][None], np.where(cls == 1, a[1][None], a[2][None]))
        valid_cls = classes[:, None, :]
        valid = np.where(valid_cls == 0, fits[0].valid[None],
                         np.where(valid_cls == 1, fits[1].valid[None], fits[2].valid[None]))
        return MultiFit(self.variables, self.months, self.day_month, pick("mean"), pick("std"),
                        self.transition, self.innovation_cholesky, self.initial_cholesky,
                        valid,
                        self.climatological_residuals, self.climatological_innovations,
                        diagnostics=dict(self.diagnostics, mixture="member-specific class curves"))


def _transform(name: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if name in ("HUMIN", "HUMAX"):
        if np.any(finite & ((values < 0) | (values > 100))):
            raise ValueError(f"{name} observations must lie in [0, 100] percent")
        # Exact endpoint observations require finite latent values. Endpoints
        # map to 0.01 and 99.99 percent and are explicitly reported below.
        transformed = logit(np.clip(values / 100.0, 1e-4, 1 - 1e-4))
    elif name in ("WIND", "SOLAR"):
        if np.any(finite & (values < 0)):
            raise ValueError(f"{name} observations must be nonnegative")
        transformed = np.log1p(values)
    else:
        transformed = values.copy()
    return np.where(finite, transformed, np.nan)


def _inverse_transform(name: str, values: np.ndarray) -> np.ndarray:
    if name in ("HUMIN", "HUMAX"):
        return 100 * expit(values)
    if name in ("WIND", "SOLAR"):
        # A transformed Gaussian can be negative; clipping creates a point
        # mass at zero. This is not a fitted hurdle or calm-wind model.
        return np.expm1(np.clip(values, 0, 50))
    return values


def _quadratic_fit(design: np.ndarray, y: np.ndarray, weights: np.ndarray,
                   target: np.ndarray) -> np.ndarray:
    """Stable weighted least squares; reduce degree if days are sparse."""
    keep = np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return np.full(target.shape[0], np.nan)
    x, z, w = design[keep], y[keep], weights[keep]
    w = w / w.max()
    degree = min(2, np.unique(x[:, 1]).size - 1)
    root = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(x[:, :degree + 1] * root[:, None],
                              z * root, rcond=None)
    return target[:, :degree + 1] @ coef


def _fit_curves(values: np.ndarray, rain: np.ndarray, weights: np.ndarray,
                wet_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one variable at one site, keeping wet/dry fallback explicit."""
    ny, nd = values.shape
    t = np.linspace(-1, 1, nd)
    daily_design = np.column_stack((np.ones(nd), t, t * t))
    design = np.tile(daily_design, (ny, 1))
    flat = values.reshape(-1)
    w = np.repeat(weights, nd)
    observed = np.isfinite(flat) & np.isfinite(rain.reshape(-1)) & (w > 0)
    states = (rain.reshape(-1) >= wet_threshold)
    mean = np.full((2, nd), np.nan)
    std = mean.copy()
    fallback = np.zeros(2, dtype=bool)
    for state in (0, 1):
        use = observed & (states == state)
        if use.sum() < 6:
            use = observed
            fallback[state] = True
        if not use.any():
            continue
        working_weights = np.where(use, w, 0.0)
        mu = _quadratic_fit(design, flat, working_weights, daily_design)
        squared = (flat - np.tile(mu, ny)) ** 2
        pooled_variance = np.average(squared[use], weights=w[use])
        if pooled_variance <= 1e-16:
            variance = np.zeros(nd)
        else:
            # Fit the log of daily weighted second moments, not a polynomial
            # on the raw variance. A raw quadratic can turn negative at a
            # seasonal edge: replacing that value by a tiny floor creates
            # enormous standardized residuals and corrupts the monthly VAR.
            daily_weight = working_weights.reshape(ny, nd).sum(axis=0)
            daily_numerator = np.where(use, squared * w, 0.0).reshape(ny, nd).sum(axis=0)
            daily_moment = np.divide(daily_numerator, daily_weight,
                                     out=np.full(nd, np.nan), where=daily_weight > 0)
            positive = np.isfinite(daily_moment) & (daily_moment > 0)
            log_moment = np.full(nd, np.nan)
            log_moment[positive] = np.log(daily_moment[positive])
            log_variance = _quadratic_fit(daily_design, log_moment,
                                          np.where(positive, daily_weight, 0), daily_design)
            # Exponentiation guarantees positive variance. Rescaling preserves
            # the pooled forecast-weighted residual second moment despite the
            # logarithm/Jensen effect; relative daily variation stays smooth.
            relative = np.exp(log_variance - np.max(log_variance))
            relative_mean = np.average(relative, weights=daily_weight)
            variance = relative * (pooled_variance / relative_mean)
        mean[state] = mu
        std[state] = np.sqrt(variance)
    return mean, std, fallback


def _safe_cholesky(covariance: np.ndarray) -> np.ndarray:
    """Symmetrize and repair a covariance with a small eigenvalue floor."""
    cov = np.asarray(covariance, dtype=float)
    cov = (cov + cov.T) / 2
    eig, vectors = np.linalg.eigh(cov)
    scale = max(float(np.max(np.abs(eig))), 1.0)
    cov = (vectors * np.maximum(eig, scale * 1e-9)) @ vectors.T
    return np.linalg.cholesky((cov + cov.T) / 2)


def _estimate_var(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ridge VAR fit with zero intercept to centered standardized residuals."""
    nvar = previous.shape[1]
    xtx = previous.T @ previous
    ridge = 0.01 * max(np.trace(xtx) / max(nvar, 1), 1.0)
    a = np.linalg.solve(xtx + ridge * np.eye(nvar), previous.T @ current).T
    radius = float(np.max(np.abs(np.linalg.eigvals(a))))
    if radius > 0.98:
        a *= 0.98 / radius
    errors = current - previous @ a.T
    cov = errors.T @ errors / max(len(errors), 1)
    # Small diagonal shrinkage keeps a nearly singular multi-variable fit
    # numerically usable without changing the fitted variable order.
    cov = 0.98 * cov + 0.02 * np.diag(np.diag(cov))
    return a, cov


def _diagonal_fallback(residual: np.ndarray, target_days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Univariate AR fallback when no adequate complete-case vector exists."""
    nvar = residual.shape[-1]
    a = np.zeros((nvar, nvar))
    q = np.zeros((nvar, nvar))
    for v in range(nvar):
        prev = residual[:, :-1, v][:, target_days].reshape(-1)
        curr = residual[:, 1:, v][:, target_days].reshape(-1)
        good = np.isfinite(prev) & np.isfinite(curr)
        if good.sum() >= 3:
            p, c = prev[good], curr[good]
            slope = np.dot(p, c) / (np.dot(p, p) + 0.01 * len(p))
            a[v, v] = np.clip(slope, -0.98, 0.98)
            q[v, v] = np.mean((c - a[v, v] * p) ** 2)
        else:
            finite = residual[..., v][np.isfinite(residual[..., v])]
            q[v, v] = np.mean(finite ** 2) if finite.size else 1.0
    return a, q


def fit_multivariate(values: dict[str, np.ndarray], rain: np.ndarray,
                     month: np.ndarray, weights: np.ndarray,
                     wet_threshold: float = 1.0, dependence: MultiFit | None = None) -> MultiFit:
    """Fit non-precipitation variables using seasonal forecast year weights.

    The same rainfall-derived weights condition all supplied variables through
    their historical association with seasonal rain; this does not introduce
    an independent humidity, wind, radiation, or temperature forecast.
    Climatological VAR fits ignore forecast weights and never connect years.
    Sparse monthly fits fall back to a seasonal VAR, then diagonal AR models.

    ``dependence`` reuses the climatological residuals and VAR(1) of an earlier
    fit (e.g. the forecast-mean fit) so that tercile-class fits only re-estimate
    the conditioned wet/dry mean and standard-deviation curves.
    """
    unknown = set(values) - set(VARIABLE_ORDER)
    if unknown:
        raise ValueError(f"Unsupported multivariate variables: {sorted(unknown)}")
    rain = np.asarray(rain, dtype=float)
    month = np.asarray(month)
    weights = np.asarray(weights, dtype=float)
    if rain.ndim != 3:
        raise ValueError("rain must have dimensions (year, day, site)")
    ny, nd, ns = rain.shape
    if ny < 1 or nd < 1 or ns < 1:
        raise ValueError("rain axes must be nonempty")
    if month.shape != (nd,) or not np.all(np.isin(month, np.arange(1, 13))):
        raise ValueError("month must contain one month number (1..12) per day")
    if weights.shape != (ny, ns):
        raise ValueError("weights must have dimensions (year, site)")
    if np.any(np.isinf(weights)) or np.any(weights < 0):
        raise ValueError("weights must be nonnegative and finite or missing")
    if not np.isfinite(wet_threshold) or wet_threshold <= 0:
        raise ValueError("wet_threshold must be finite and positive")
    if np.any(np.isfinite(rain) & (rain < 0)):
        raise ValueError("rain observations must be nonnegative")
    weights = np.where(np.isfinite(weights), weights, 0)
    largest = weights.max(axis=0)
    weights = np.divide(weights, largest, out=np.zeros_like(weights), where=largest > 0)
    variables = tuple(v for v in VARIABLE_ORDER if v in values)
    for v in variables:
        if np.shape(values[v]) != rain.shape:
            raise ValueError(f"{v} must have the same shape as rain")
    for lo, hi in (("TMIN", "TMAX"), ("HUMIN", "HUMAX")):
        if lo in values and hi in values:
            lower, upper = np.asarray(values[lo]), np.asarray(values[hi])
            if np.any(np.isfinite(lower) & np.isfinite(upper) & (lower > upper)):
                raise ValueError(f"Historical {lo} must not exceed {hi}")
    months = np.array(list(dict.fromkeys(month.tolist())), dtype=int)
    nv, nm = len(variables), len(months)
    mean = np.full((nv, 2, nd, ns), np.nan)
    std = mean.copy()
    valid = np.zeros((nv, ns), dtype=bool)
    residuals = {v: np.full(rain.shape, np.nan) for v in variables}
    innovations = {v: np.full(rain.shape, np.nan) for v in variables}
    fallback = np.zeros((nv, 2, ns), dtype=bool)
    transformed = {v: _transform(v, values[v]) for v in variables}
    for iv, variable in enumerate(variables):
        for site in range(ns):
            v = transformed[variable][:, :, site]
            r = rain[:, :, site]
            mu, sigma, used_fallback = _fit_curves(v, r, weights[:, site], wet_threshold)
            mean[iv, :, :, site], std[iv, :, :, site] = mu, sigma
            fallback[iv, :, site] = used_fallback
            valid[iv, site] = np.all(np.isfinite(mu)) & np.all(np.isfinite(sigma))
            if dependence is not None:
                continue
            cmu, csigma, _ = _fit_curves(v, r, np.ones(ny), wet_threshold)
            states = (r >= wet_threshold).astype(int)
            observed_mean = np.take_along_axis(cmu, states, axis=0)
            observed_std = np.take_along_axis(csigma, states, axis=0)
            z = np.divide(v - observed_mean, observed_std,
                          out=np.zeros_like(v), where=observed_std > 1e-8)
            z[~(np.isfinite(v) & np.isfinite(r) & np.isfinite(observed_mean))] = np.nan
            residuals[variable][:, :, site] = z

    if dependence is not None:
        if (tuple(dependence.variables) != variables or dependence.transition.shape[:2] != (nm, ns)
                or not np.array_equal(dependence.months, months)
                or not np.array_equal(dependence.day_month, month)):
            raise ValueError("dependence fit must have the same variables, months and sites")
        return MultiFit(
            variables, months, month.astype(int), mean, std, dependence.transition,
            dependence.innovation_cholesky, dependence.initial_cholesky, valid,
            dependence.climatological_residuals, dependence.climatological_innovations,
            diagnostics=dict(dependence.diagnostics, curve_state_fallback=fallback))

    transition = np.zeros((nm, ns, nv, nv))
    innovation_cholesky = np.zeros_like(transition)
    initial_cholesky = np.zeros_like(transition)
    var_source = np.full((nm, ns), "unavailable", dtype="U16")
    pair_count = np.zeros((nm, ns), dtype=int)
    for site in range(ns):
        if not nv:
            continue
        series = np.stack([residuals[v][:, :, site] for v in variables], axis=-1)
        active = np.flatnonzero(np.any(np.isfinite(series), axis=(0, 1)))
        if not active.size:
            continue
        z = series[:, :, active]
        previous = z[:, :-1].reshape(-1, len(active))
        current = z[:, 1:].reshape(-1, len(active))
        complete = np.all(np.isfinite(previous) & np.isfinite(current), axis=1)
        month_for_pair = np.tile(month[1:], ny)
        for im, mon in enumerate(months):
            use = complete & (month_for_pair == mon)
            source = "monthly"
            if use.sum() < max(24, 4 * len(active)):
                use = complete
                source = "seasonal"
            if use.sum() >= max(12, 2 * len(active)):
                a, q = _estimate_var(previous[use], current[use])
            else:
                a, q = _diagonal_fallback(z, np.ones(max(nd - 1, 0), dtype=bool))
                source = "diagonal_AR"
            pair_count[im, site] = int(use.sum())
            var_source[im, site] = source
            l = _safe_cholesky(q)
            stationary = solve_discrete_lyapunov(a, l @ l.T)
            c = _safe_cholesky(stationary)
            idx = np.ix_(active, active)
            transition[im, site][idx] = a
            innovation_cholesky[im, site][idx] = l
            initial_cholesky[im, site][idx] = c
            # Whiten in local active-variable order, exactly matching the
            # simulation's Cholesky streams; retain gaps and first-day NaN.
            for day in np.flatnonzero(month == mon):
                if day == 0:
                    continue
                prev, curr = z[:, day - 1], z[:, day]
                good = np.all(np.isfinite(prev) & np.isfinite(curr), axis=1)
                if not good.any():
                    continue
                whitened = np.linalg.solve(l, (curr[good] - prev[good] @ a.T).T).T
                for ia, iv in enumerate(active):
                    innovations[variables[iv]][good, day, site] = whitened[:, ia]

    return MultiFit(
        variables, months, month.astype(int), mean, std, transition,
        innovation_cholesky, initial_cholesky, valid, residuals, innovations,
        diagnostics={
            "curve_state_fallback": fallback,
            "var_source": var_source,
            "var_pair_count": pair_count,
            "transformations": {v: ("logit(RH/100), endpoint epsilon=1e-4" if v.startswith("HUM")
                                      else "log1p; inverse clipped at zero" if v in ("WIND", "SOLAR")
                                      else "identity (degrees Celsius)") for v in variables},
            "variance_method": "weighted quadratic log-variance fitted to daily residual second moments, "
                               "rescaled to pooled weighted variance; constants remain deterministic",
            "dependence": "unweighted climatological VAR(1); ridge=0.01; spectral radius <=0.98",
            "limitations": "Humidity/wind/solar are extensions. Pair sorting changes marginals. "
                           "No astronomical radiation upper bound. Spatial kernels should be "
                           "calibrated on whitened innovations; distance fits remain approximate.",
        },
    )


def simulate_multivariate(fit: MultiFit, rain_generated: np.ndarray,
                          month: np.ndarray, normal_draw: Callable,
                          wet_threshold: float = 1.0) -> dict[str, np.ndarray]:
    """Simulate weather with supplied spatial normal innovations.

    ``normal_draw(n_members, step, stream)`` must return unit normal fields of
    shape (member, site), with independent streams numbered 10, 11, ... in
    ``fit.variables`` order. Initial values use step=-1 and the first month's
    stationary covariance. Seasonal years are independent realizations; the
    caller must not join their first/last days to estimate transition skill.
    """
    rain = np.asarray(rain_generated, dtype=float)
    month = np.asarray(month)
    if rain.ndim != 3:
        raise ValueError("rain_generated must have dimensions (member, day, site)")
    n_members, nd, ns = rain.shape
    member_axis = fit.mean.ndim == 5
    if member_axis and fit.mean.shape[0] != n_members:
        raise ValueError("Member-specific fit parameters must match the generated member count")
    members, sites = np.arange(n_members)[:, None], np.arange(ns)[None, :]
    if (nd, ns) != fit.mean.shape[-2:] or not np.array_equal(month, fit.day_month):
        raise ValueError("Simulation days, sites and month sequence must match the fit")
    if not np.isfinite(wet_threshold) or wet_threshold <= 0:
        raise ValueError("wet_threshold must be finite and positive")
    if np.any(np.isfinite(rain) & (rain < 0)):
        raise ValueError("rain_generated must be nonnegative")
    nv = len(fit.variables)
    if nv == 0:
        return {}

    def draw(step: int) -> np.ndarray:
        fields = []
        for iv in range(nv):
            field_values = np.asarray(normal_draw(n_members, step, 10 + iv), dtype=float)
            if field_values.shape != (n_members, ns) or not np.all(np.isfinite(field_values)):
                raise ValueError("normal_draw must return finite (member, site) normal fields")
            fields.append(field_values)
        return np.stack(fields, axis=-1)

    result = {v: np.full(rain.shape, np.nan) for v in fit.variables}
    lookup = {int(m): i for i, m in enumerate(fit.months)}
    first = lookup[int(month[0])]
    state = np.einsum("sij,msj->msi", fit.initial_cholesky[first], draw(-1))
    for day, mon in enumerate(month):
        im = lookup[int(mon)]
        if day > 0:
            state = (np.einsum("sij,msj->msi", fit.transition[im], state)
                     + np.einsum("sij,msj->msi", fit.innovation_cholesky[im], draw(day)))
        wet = (rain[:, day] >= wet_threshold).astype(int)
        for iv, variable in enumerate(fit.variables):
            if member_axis:
                mu = fit.mean[members, iv, wet, day, sites]
                sd = fit.std[members, iv, wet, day, sites]
            else:
                mu = fit.mean[iv, wet, day, sites]
                sd = fit.std[iv, wet, day, sites]
            physical = _inverse_transform(variable, mu + sd * state[:, :, iv])
            fitted_valid = fit.valid[:, iv, :] if fit.valid.ndim == 3 else fit.valid[iv][None, :]
            valid = fitted_valid & np.isfinite(rain[:, day])
            result[variable][:, day] = np.where(valid, physical, np.nan)
    crossing_counts = {}
    for lo, hi in (("TMIN", "TMAX"), ("HUMIN", "HUMAX")):
        if lo not in result or hi not in result:
            continue
        lower, upper = result[lo], result[hi]
        crossed = np.isfinite(lower) & np.isfinite(upper) & (lower > upper)
        crossing_counts[f"{lo}_{hi}_swaps"] = int(crossed.sum())
        result[lo] = np.where(crossed, upper, lower)
        result[hi] = np.where(crossed, lower, upper)
    fit.simulation_diagnostics = crossing_counts
    return result
