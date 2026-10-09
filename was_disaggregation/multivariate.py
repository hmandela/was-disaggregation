"""Forecast-conditioned multivariate weather accompanying generated rainfall.

This is a Richardson-style extension of Wilks (2002). Forecast year weights
affect wet/dry mean and dispersion curves; the Wilks temperature option uses
affine mean and quadratic standard-deviation surfaces in forecast probability
and quadratic date fits. A separate ``moment`` option uses daily class-mixture
moments without that surface approximation. The local VAR(1) and spatial
dependence are fitted from unweighted historical residuals.
Humidity, wind and radiation are extensions, not an exact reproduction of that
paper. In particular, humidity uses a logit transform, and wind and solar use
log1p. Solar radiation has no astronomical upper bound in this implementation.

Array conventions are (year, day, site) for training and (member, day, site)
for simulation. Temperatures must already be in degrees Celsius, humidity in
percent, wind in m/s, and solar radiation in the caller's nonnegative unit.

Scientific attribution
----------------------
* Richardson, C. W. (1981), "Stochastic simulation of daily precipitation,
  temperature, and solar radiation", Water Resources Research 17(1), 182-190,
  https://doi.org/10.1029/WR017i001p00182: temperature/radiation conditioned
  on rainfall occurrence and a multivariate residual time process.
* Wilks, D. S. (2002), "Realizations of daily weather in forecast seasonal
  climate", Journal of Hydrometeorology 3(2), 195-207,
  https://doi.org/10.1175/1525-7541(2002)003<0195:RODWIF>2.0.CO;2:
  section 4b, Equations (10)-(12), independent seasonal-temperature
  conditioning of wet/dry means and standard deviations; climatological
  residual dependence is held fixed as an approximation.

The ridge VAR fit, stability adjustment and optional stationary covariance
normalization are package extensions. Normalization solves the discrete
Lyapunov equation and enforces unit marginal residual variance; it does not
recover Richardson's original coefficient estimates. Variable-specific
Gaussian innovation fields are a spatial approximation: they do not reproduce
the full between-site VAR of Wilks or guarantee physical cross-correlations.
Humidity transforms, wind/radiation transforms and pair-order repair are also
package choices. Sorting TMIN/TMAX or HUMIN/HUMAX changes their marginals;
the ``pair_policy`` parameter makes that tradeoff explicit. No combination
establishes reproduction of the authors' original station experiments.
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
        if any(f.variables != self.variables or f.valid.shape != self.valid.shape
               or not np.array_equal(f.day_month, self.day_month)
               or not np.array_equal(f.months, self.months)
               or any(not np.array_equal(getattr(f, key), getattr(self, key), equal_nan=True)
                      for key in ("transition", "innovation_cholesky", "initial_cholesky"))
               for f in fits):
            raise ValueError("class fits must share variables, dates and climatological dependence")
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
                wet_threshold: float, wet_rule: str = "ge") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one variable at one site, keeping wet/dry fallback explicit."""
    ny, nd = values.shape
    t = np.linspace(-1, 1, nd)
    daily_design = np.column_stack((np.ones(nd), t, t * t))
    design = np.tile(daily_design, (ny, 1))
    flat = values.reshape(-1)
    w = np.repeat(weights, nd)
    observed = np.isfinite(flat) & np.isfinite(rain.reshape(-1)) & (w > 0)
    states = (rain.reshape(-1) >= wet_threshold if wet_rule == "ge"
              else rain.reshape(-1) > wet_threshold)
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


def _temperature_surfaces(values: np.ndarray, rain: np.ndarray, categories: np.ndarray,
                          probabilities: np.ndarray, *, wet_threshold: float,
                          wet_rule: str, surface: str, grid_resolution: int):
    """Wilks (2002), Eq. (10) affine mean and Eq. (12) quadratic SD surfaces.

    Conditional daily class moments are population moments under the
    empirical year-resampling distribution. Each class contributes its
    forecast probability even if historical class counts differ. The total
    variance includes within-class and between-class terms; replacing it by
    the weighted average of class standard deviations would be incorrect.
    Date smoothing and the probability-grid resolution are numerical choices;
    ``surface='moment'`` is a package extension using unsmoothed daily moments.
    """
    from .conditioning import year_weights, FLAG_INVALID_PROBABILITY

    ny, nd, ns = values.shape
    categories = np.asarray(categories)
    probabilities = np.asarray(probabilities, dtype=float)
    if categories.shape != (ny, ns) or probabilities.shape != (3, ns):
        raise ValueError("temperature forecasts require (year, site) categories and (3, site) probabilities")
    _, flags = year_weights(categories, probabilities, empty_policy="renormalize")
    total = probabilities.sum(axis=0)
    p = np.divide(probabilities, total, out=np.full_like(probabilities, np.nan),
                  where=np.isfinite(total)[None] & (total[None] > 0))
    supported = np.stack([(categories == c).any(axis=0) for c in range(3)]).all(axis=0)
    good_site = supported & ((flags & FLAG_INVALID_PROBABILITY) == 0)
    t = np.linspace(-1, 1, nd)
    date_design = np.column_stack((np.ones(nd), t, t * t))
    grid = np.array([(i, j, grid_resolution - i - j)
                     for i in range(grid_resolution + 1)
                     for j in range(grid_resolution - i + 1)], float) / grid_resolution
    grid_design = np.column_stack((np.ones(len(grid)), grid[:, 0], grid[:, 2],
                                   grid[:, 0] ** 2, grid[:, 0] * grid[:, 2], grid[:, 2] ** 2))
    mean = np.full((2, nd, ns), np.nan)
    std = mean.copy()
    mean_coefficients = np.full((2, ns, 3, 3), np.nan)  # state,site,class,date coefficient
    sd_coefficients = np.full((2, ns, 6, 3), np.nan)  # state,site,forecast monomial,date coefficient
    fallback = np.zeros((2, ns), bool)
    sd_projected = np.zeros((2, nd, ns), bool)
    mean_surface_residual = np.full((2, ns), np.nan)
    variance_between_class = np.full((2, nd, ns), np.nan)
    observed = np.isfinite(values) & np.isfinite(rain)
    state = (rain >= wet_threshold if wet_rule == "ge" else rain > wet_threshold)
    for site in np.flatnonzero(good_site):
        for wet_state in (0, 1):
            first = np.full((3, nd), np.nan)
            class_variance = first.copy()
            for c in range(3):
                in_class = categories[:, site] == c
                mask = observed[:, :, site] & (state[:, :, site] == wet_state) & in_class[:, None]
                count = mask.sum(axis=0)
                class_sample = values[in_class, :, site]
                overall = np.isfinite(class_sample) & np.isfinite(rain[in_class, :, site])
                # Empty class/state/date: fall back only to that same class's
                # unconditional daily moment, and record the fallback.
                missing = count == 0
                if missing.any():
                    fallback[wet_state, site] = True
                    full_mask = observed[:, :, site] & in_class[:, None]
                    mask[:, missing] = full_mask[:, missing]
                    count = mask.sum(axis=0)
                first[c] = np.divide(np.where(mask, values[:, :, site], 0).sum(axis=0), count,
                                      out=np.full(nd, np.nan), where=count > 0)
                # Centred differences retain exact zero spread for identical
                # observations, unlike subtracting two large raw moments.
                centred = values[:, :, site] - first[c][None, :]
                class_variance[c] = np.divide(np.where(mask, centred ** 2, 0).sum(axis=0), count,
                                              out=np.full(nd, np.nan), where=count > 0)
                # Persistent daily gaps can be estimated from the seasonal
                # class sample, but missing an entire class invalidates site.
                if not np.isfinite(first[c]).any() or not overall.any():
                    good_site[site] = False
                    break
                fitted = _quadratic_fit(date_design, first[c], np.isfinite(first[c]).astype(float), date_design)
                coefficients = np.linalg.lstsq(date_design, fitted, rcond=None)[0]
                mean_coefficients[wet_state, site, c] = coefficients
                absent = ~np.isfinite(first[c])
                if absent.any():
                    fallback[wet_state, site] = True
                    pooled = np.nanvar(class_sample)
                    first[c, absent] = fitted[absent]
                    class_variance[c, absent] = pooled
            if not good_site[site]:
                continue
            class_curves = mean_coefficients[wet_state, site] @ date_design.T
            target_mean = p[:, site] @ class_curves
            mean[wet_state, :, site] = target_mean
            raw_mean = p[:, site] @ first
            variance_between_class[wet_state, :, site] = p[:, site] @ (first - raw_mean) ** 2
            mean_surface_residual[wet_state, site] = float(np.sqrt(np.mean((target_mean - raw_mean) ** 2)))
            if surface == "moment":
                # Exact population mixture moments require the raw daily
                # mean. A residual MSE around a smoothed mean is not variance
                # and would invent spread for deterministic seasonal cycles.
                mean[wet_state, :, site] = raw_mean
                variance = p[:, site] @ class_variance + variance_between_class[wet_state, :, site]
                std[wet_state, :, site] = np.sqrt(np.maximum(variance, 0.))
                continue
            grid_date_coefficients = np.zeros((len(grid), 3))
            for k, triple in enumerate(grid):
                mu = triple @ class_curves
                v = triple @ class_variance + triple @ (first - mu) ** 2
                sd = np.sqrt(np.maximum(v, 0.))
                grid_date_coefficients[k] = np.linalg.lstsq(date_design, sd, rcond=None)[0]
            forecast_coefficients = np.linalg.lstsq(grid_design, grid_date_coefficients, rcond=None)[0]
            sd_coefficients[wet_state, site] = forecast_coefficients
            b, a = p[0, site], p[2, site]
            target_design = np.array([1., b, a, b * b, b * a, a * a])
            predicted_sd = date_design @ (target_design @ forecast_coefficients)
            sd_projected[wet_state, :, site] = predicted_sd < 0
            std[wet_state, :, site] = np.maximum(predicted_sd, 0.)
    mean[:, :, ~good_site] = np.nan
    std[:, :, ~good_site] = np.nan
    return mean, std, good_site, {
        "surface": surface, "forecast_flags": flags, "missing_tercile": ~supported,
        "class_state_fallback": fallback, "mean_date_class_coefficients": mean_coefficients,
        "std_forecast_date_coefficients": sd_coefficients, "std_projected": sd_projected,
        "forecast_grid": grid, "raw_mean_smoothing_rmse": mean_surface_residual,
        "between_class_daily_variance": variance_between_class,
        "variance_estimand": ("population variance of empirical class-conditioned resampling"
                              if surface == "moment" else
                              "quadratic forecast/date approximation of residual MSE around smoothed class means"),
    }


def fit_multivariate(values: dict[str, np.ndarray], rain: np.ndarray,
                     month: np.ndarray, weights: np.ndarray,
                     wet_threshold: float = 1.0, dependence: MultiFit | None = None,
                     *, variable_weights: dict[str, np.ndarray] | None = None,
                     wet_rule: str = "ge",
                     temperature_forecasts: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
                     temperature_surface: str = "wilks_2002",
                     temperature_grid_resolution: int = 4,
                     residual_covariance: str = "unit") -> MultiFit:
    """Fit non-precipitation variables using seasonal forecast year weights.

    By default rainfall-derived weights condition all supplied variables
    through their historical association with seasonal rain. Optional
    ``variable_weights`` maps a variable name to independent (year, site)
    weights for its wet/dry mean and standard-deviation curves. This permits
    independent seasonal temperature forecasts to condition TMIN and TMAX,
    while preserving the climatological joint residual process.
    Climatological VAR fits ignore forecast weights and never connect years.
    Sparse monthly fits fall back to a seasonal VAR, then diagonal AR models.

    ``dependence`` reuses the climatological residuals and VAR(1) of an earlier
    fit (e.g. the forecast-mean fit) so that tercile-class fits only re-estimate
    the conditioned wet/dry mean and standard-deviation curves.

    ``temperature_forecasts`` maps observed TMIN/TMAX to a pair of categories
    (year, site) and forecast probabilities (3, site). ``temperature_surface``
    selects Wilks's (2002), section 4b, affine/quadratic forecast and date
    approximations (full citation in this module docstring), or
    ``'moment'`` for raw daily mixture means and exact population variances
    without date smoothing. Daily missing-state/class fallbacks are reported;
    a wholly absent seasonal class invalidates that temperature site.
    """
    unknown = set(values) - set(VARIABLE_ORDER)
    if unknown:
        raise ValueError(f"Unsupported multivariate variables: {sorted(unknown)}")
    if temperature_surface not in {"wilks_2002", "moment"}:
        raise ValueError("temperature_surface must be 'wilks_2002' or 'moment'")
    if isinstance(temperature_grid_resolution, (bool, np.bool_)) \
            or not isinstance(temperature_grid_resolution, (int, np.integer)) \
            or not 2 <= temperature_grid_resolution <= 12:
        raise ValueError("temperature_grid_resolution must be an integer from 2 to 12")
    if residual_covariance not in {"unit", "fit"}:
        raise ValueError("residual_covariance must be 'unit' or 'fit'")
    temperature_forecasts = {} if temperature_forecasts is None else temperature_forecasts
    if not isinstance(temperature_forecasts, dict) \
            or set(temperature_forecasts) - (set(values) & {"TMIN", "TMAX"}):
        raise ValueError("temperature_forecasts must map observed TMIN/TMAX to (categories, probabilities)")
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
    if not np.isfinite(wet_threshold) or wet_threshold < 0 or (wet_threshold == 0 and wet_rule != "gt"):
        raise ValueError("wet_threshold must be positive, or zero with wet_rule='gt'")
    if wet_rule not in {"ge", "gt"}:
        raise ValueError("wet_rule must be 'ge' or 'gt'")
    if np.isinf(rain).any() or np.any(np.isfinite(rain) & (rain < 0)):
        raise ValueError("rain observations must be nonnegative and finite, or NaN for missing")
    weights = np.where(np.isfinite(weights), weights, 0)
    largest = weights.max(axis=0)
    weights = np.divide(weights, largest, out=np.zeros_like(weights), where=largest > 0)
    variables = tuple(v for v in VARIABLE_ORDER if v in values)
    variable_weights = {} if variable_weights is None else variable_weights
    if not isinstance(variable_weights, dict) or set(variable_weights) - set(variables):
        raise ValueError("variable_weights must map observed variables to (year, site) weights")
    local_weights = {}
    fallback_weight_sites = {}
    for variable, array in variable_weights.items():
        array = np.asarray(array, dtype=float)
        if array.shape != (ny, ns) or np.any(~np.isfinite(array)) or np.any(array < 0):
            raise ValueError(f"variable_weights[{variable!r}] must be finite nonnegative (year, site) weights")
        maximum = array.max(axis=0)
        fallback_weight_sites[variable] = maximum <= 0
        scaled = np.divide(array, maximum, out=np.zeros_like(array), where=maximum > 0)
        # Masked temperature forecast at an otherwise valid rainfall site:
        # retain the legacy rainfall-conditioned curve there, while reporting
        # precisely where the independent forecast was unavailable.
        local_weights[variable] = np.where((maximum > 0)[None], scaled, weights)
    for v in variables:
        if np.shape(values[v]) != rain.shape:
            raise ValueError(f"{v} must have the same shape as rain")
        if np.isinf(np.asarray(values[v], dtype=float)).any():
            raise ValueError(f"{v} must be finite, or NaN for missing")
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
    temperature_diagnostics = {}
    for iv, variable in enumerate(variables):
        for site in range(ns):
            v = transformed[variable][:, :, site]
            r = rain[:, :, site]
            mu, sigma, used_fallback = _fit_curves(v, r, local_weights.get(variable, weights)[:, site],
                                                   wet_threshold, wet_rule)
            mean[iv, :, :, site], std[iv, :, :, site] = mu, sigma
            fallback[iv, :, site] = used_fallback
            valid[iv, site] = np.all(np.isfinite(mu)) & np.all(np.isfinite(sigma))
            if dependence is not None:
                continue
            cmu, csigma, _ = _fit_curves(v, r, np.ones(ny), wet_threshold, wet_rule)
            states = (r >= wet_threshold if wet_rule == "ge" else r > wet_threshold).astype(int)
            observed_mean = np.take_along_axis(cmu, states, axis=0)
            observed_std = np.take_along_axis(csigma, states, axis=0)
            z = np.divide(v - observed_mean, observed_std,
                          out=np.zeros_like(v), where=observed_std > 1e-8)
            z[~(np.isfinite(v) & np.isfinite(r) & np.isfinite(observed_mean))] = np.nan
            residuals[variable][:, :, site] = z

        if variable in temperature_forecasts:
            forecast = temperature_forecasts[variable]
            if not isinstance(forecast, (tuple, list)) or len(forecast) != 2:
                raise ValueError("temperature_forecasts entries must contain (categories, probabilities)")
            mu, sigma, surface_valid, diagnostic = _temperature_surfaces(
                transformed[variable], rain, *forecast,
                wet_threshold=wet_threshold, wet_rule=wet_rule,
                surface=temperature_surface, grid_resolution=temperature_grid_resolution)
            mean[iv], std[iv], valid[iv] = mu, sigma, surface_valid
            temperature_diagnostics[variable] = diagnostic

    if dependence is not None:
        if (tuple(dependence.variables) != variables or dependence.transition.shape[:2] != (nm, ns)
                or not np.array_equal(dependence.months, months)
                or not np.array_equal(dependence.day_month, month)):
            raise ValueError("dependence fit must have the same variables, months and sites")
        return MultiFit(
            variables, months, month.astype(int), mean, std, dependence.transition,
            dependence.innovation_cholesky, dependence.initial_cholesky, valid,
            dependence.climatological_residuals, dependence.climatological_innovations,
            diagnostics=dict(dependence.diagnostics, curve_state_fallback=fallback,
                             variable_specific_weights=tuple(sorted(local_weights)),
                             variable_weight_fallback_sites=fallback_weight_sites,
                             temperature_surfaces=temperature_diagnostics))

    transition = np.zeros((nm, ns, nv, nv))
    innovation_cholesky = np.zeros_like(transition)
    initial_cholesky = np.zeros_like(transition)
    var_source = np.full((nm, ns), "unavailable", dtype="U16")
    pair_count = np.zeros((nm, ns), dtype=int)
    stationary_marginal_std = np.full((nm, ns, nv), np.nan)
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
            stationary_std = np.sqrt(np.maximum(np.diag(stationary), 1e-12))
            stationary_marginal_std[im, site, active] = stationary_std
            original_a, original_l = a.copy(), l.copy()
            if residual_covariance == "unit":
                a = a * stationary_std[None, :] / stationary_std[:, None]
                l = l / stationary_std[:, None]
                stationary = stationary / stationary_std[:, None] / stationary_std[None, :]
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
                whitened = np.linalg.solve(original_l, (curr[good] - prev[good] @ original_a.T).T).T
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
            "variable_specific_weights": tuple(sorted(local_weights)),
            "variable_weight_fallback_sites": fallback_weight_sites,
            "wet_rule": wet_rule,
            "temperature_surfaces": temperature_diagnostics,
            "residual_covariance": residual_covariance,
            "pre_normalization_stationary_marginal_std": stationary_marginal_std,
            "limitations": "Humidity/wind/solar are extensions. Pair sorting changes marginals. "
                           "No astronomical radiation upper bound. Spatial kernels should be "
                           "calibrated on whitened innovations; distance fits remain approximate.",
        },
    )


def simulate_multivariate(fit: MultiFit, rain_generated: np.ndarray,
                          month: np.ndarray, normal_draw: Callable,
                          wet_threshold: float = 1.0, *, wet_rule: str = "ge",
                          pair_policy: str = "sort") -> dict[str, np.ndarray]:
    """Simulate weather with supplied spatial normal innovations.

    ``normal_draw(n_members, step, stream)`` must return unit normal fields of
    shape (member, site), with independent streams numbered 10, 11, ... in
    ``fit.variables`` order. Initial values use step=-1 and the first month's
    stationary covariance. Seasonal years are independent realizations; the
    caller must not join their first/last days to estimate transition skill.
    ``pair_policy='sort'`` preserves the legacy enforcement of TMIN<=TMAX
    and HUMIN<=HUMAX, while reporting both counts and fractions of swaps.
    Sorting alters both original marginal distributions. ``'none'`` retains
    the Gaussian-model output for scientific diagnostics; ``'error'`` rejects
    any crossing instead of altering it. Independent marginal temperature
    forecasts need not define a jointly physically feasible distribution.
    """
    rain = np.asarray(rain_generated, dtype=float)
    if pair_policy not in {"sort", "none", "error"}:
        raise ValueError("pair_policy must be 'sort', 'none' or 'error'")
    month = np.asarray(month)
    if rain.ndim != 3 or not all(rain.shape):
        raise ValueError("rain_generated must have nonempty dimensions (member, day, site)")
    n_members, nd, ns = rain.shape
    member_axis = fit.mean.ndim == 5
    if member_axis and fit.mean.shape[0] != n_members:
        raise ValueError("Member-specific fit parameters must match the generated member count")
    members, sites = np.arange(n_members)[:, None], np.arange(ns)[None, :]
    if (nd, ns) != fit.mean.shape[-2:] or not np.array_equal(month, fit.day_month):
        raise ValueError("Simulation days, sites and month sequence must match the fit")
    if not np.isfinite(wet_threshold) or wet_threshold < 0 or (wet_threshold == 0 and wet_rule != "gt"):
        raise ValueError("wet_threshold must be positive, or zero with wet_rule='gt'")
    if wet_rule not in {"ge", "gt"}:
        raise ValueError("wet_rule must be 'ge' or 'gt'")
    if np.isinf(rain).any() or np.any(np.isfinite(rain) & (rain < 0)):
        raise ValueError("rain_generated must be nonnegative and finite, or NaN for missing")
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
    state_covariance = np.einsum("sij,skj->sik", fit.initial_cholesky[first], fit.initial_cholesky[first])
    normalization_range = [np.inf, 0.]
    for day, mon in enumerate(month):
        im = lookup[int(mon)]
        if day > 0:
            state = (np.einsum("sij,msj->msi", fit.transition[im], state)
                     + np.einsum("sij,msj->msi", fit.innovation_cholesky[im], draw(day)))
            a, l = fit.transition[im], fit.innovation_cholesky[im]
            state_covariance = (np.einsum("sij,sjk,slk->sil", a, state_covariance, a)
                                + np.einsum("sij,skj->sik", l, l))
        latent = state
        if fit.diagnostics.get("residual_covariance") == "unit":
            # Different monthly covariance matrices induce a transient when
            # their VARs are continued across boundaries. Propagate the actual
            # covariance to keep each output residual's variance equal to 1.
            day_std = np.sqrt(np.maximum(np.diagonal(state_covariance, axis1=-2, axis2=-1), 1e-12))
            latent = state / day_std[None]
            positive = day_std > 1e-6
            if positive.any():
                normalization_range[0] = min(normalization_range[0], float(day_std[positive].min()))
                normalization_range[1] = max(normalization_range[1], float(day_std[positive].max()))
        wet = (rain[:, day] >= wet_threshold if wet_rule == "ge"
               else rain[:, day] > wet_threshold).astype(int)
        for iv, variable in enumerate(fit.variables):
            if member_axis:
                mu = fit.mean[members, iv, wet, day, sites]
                sd = fit.std[members, iv, wet, day, sites]
            else:
                mu = fit.mean[iv, wet, day, sites]
                sd = fit.std[iv, wet, day, sites]
            physical = _inverse_transform(variable, mu + sd * latent[:, :, iv])
            fitted_valid = fit.valid[:, iv, :] if fit.valid.ndim == 3 else fit.valid[iv][None, :]
            valid = fitted_valid & np.isfinite(rain[:, day])
            result[variable][:, day] = np.where(valid, physical, np.nan)
    crossing_counts = {}
    for lo, hi in (("TMIN", "TMAX"), ("HUMIN", "HUMAX")):
        if lo not in result or hi not in result:
            continue
        lower, upper = result[lo], result[hi]
        crossed = np.isfinite(lower) & np.isfinite(upper) & (lower > upper)
        crossing_counts[f"{lo}_{hi}_crossings"] = int(crossed.sum())
        finite = np.isfinite(lower) & np.isfinite(upper)
        crossing_counts[f"{lo}_{hi}_crossing_fraction"] = float(crossed.sum() / max(finite.sum(), 1))
        crossing_counts[f"{lo}_{hi}_swaps"] = int(crossed.sum()) if pair_policy == "sort" else 0
        if pair_policy == "error" and crossed.any():
            raise ValueError(f"Generated {lo}>{hi}; joint forecast is incompatible with pair_policy='error'")
        if pair_policy == "sort":
            result[lo] = np.where(crossed, upper, lower)
            result[hi] = np.where(crossed, lower, upper)
    crossing_counts["pair_policy"] = pair_policy
    crossing_counts["latent_marginal_std_range_before_daily_normalization"] = normalization_range
    fit.simulation_diagnostics = crossing_counts
    return result
