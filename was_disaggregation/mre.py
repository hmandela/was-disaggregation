"""Multi-constraint historical year weights: minimum relative entropy and Croley.

Given K season attributes (total, onset, dry spells, ...) each with a tercile
forecast, find year weights w (per site) that

    minimise   D(w, w0) + sum_k sum_c (P_w[class_k = c] - p_kc)^2 / (2 tau_k^2)
    subject to w >= 0, sum w = 1,

where w0 is a prior (uniform, or e.g. ENSO rank weights) and

    D = KL(w || w0)                      method='mre'   (Weijs & van de Giesen 2013)
    D = (n/2) sum (w - w0)^2             method='croley' (Croley 1996, 2000)

The tolerance tau_k is the probability error you accept on constraint k: with
compatible forecasts every constraint is met to within ~tau_k^2 * lambda; with
conflicting forecasts (e.g. wet total AND late onset when history says late
onsets are dry) the solution trades the constraints off in proportion to 1/tau_k^2
instead of failing. Only the below (first) and above (last) class probabilities
enter explicitly; the middle class follows from sum w = 1.

MRE solution: w_y = w0_y exp(lambda . g_y) / Z with g_y the class indicators.
Both methods are solved in the dual by batched damped Newton over all sites.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.special import logsumexp

from .conditioning import classify_seasons, FLAG_INVALID_PROBABILITY, FLAG_NO_HISTORY

FLAG_CONSTRAINT_MISSED = 8
FLAG_LOW_EFFECTIVE_SIZE = 16

_LABEL_ALIASES = {
    "PB": "PB", "BN": "PB", "BELOW": "PB", "EARLY": "PB", "SHORT": "PB", "LOW": "PB", "PRECOCE": "PB",
    "PN": "PN", "NN": "PN", "NORMAL": "PN", "NEAR": "PN", "NORMALE": "PN",
    "PA": "PA", "AN": "PA", "ABOVE": "PA", "LATE": "PA", "LONG": "PA", "HIGH": "PA", "TARDIVE": "PA", "LONGUE": "PA",
}


def relabel_probabilities(probabilities: xr.DataArray) -> xr.DataArray:
    """Map early/normal/late, short/normal/long, BN/NN/AN ... onto PB/PN/PA.

    Convention: the first class is always the LOW attribute value (below-normal
    total, EARLY onset/cessation, SHORT dry spell), the last the HIGH value.
    """
    if "probability" not in probabilities.dims:
        dims = [d for d in probabilities.dims if d not in ("T", "Y", "X", "lat", "lon", "latitude", "longitude", "time")]
        if len(dims) != 1:
            raise ValueError("cannot identify the category dimension")
        probabilities = probabilities.rename({dims[0]: "probability"})
    labels = [str(v.decode() if isinstance(v, bytes) else v).strip().upper() for v in probabilities.probability.values]
    try:
        mapped = [_LABEL_ALIASES[l] for l in labels]
    except KeyError as err:
        raise ValueError(f"unknown category label {err}; use PB/PN/PA or early/normal/late, short/normal/long") from None
    return probabilities.assign_coords(probability=mapped)


@dataclass
class SeasonalConstraint:
    """A tercile forecast for one season attribute.

    ``probabilities``: DataArray (probability, Y, X[, T]) ordered low -> high
    (PB/PN/PA, or early/normal/late, short/normal/long). ``tolerance`` is tau.
    ``probabilities=None`` means: use the generator's main forecast (total only).
    ``thresholds`` (optional): fixed class limits (low, high) — two numbers, or
    a DataArray (threshold=2, Y, X) — instead of reference-period terciles.
    Classes are x <= low, low < x <= high, x > high. Use them when the forecast
    product defines its categories in days (e.g. dry spells <= 7, 8-14, > 14),
    or when heavy ties make empirical terciles unequal (a 'short' class
    holding 50 % of years).
    """
    attribute: object
    probabilities: xr.DataArray | None = None
    tolerance: float = 0.01
    name: str | None = None
    thresholds: object = None

    def __post_init__(self):
        if self.probabilities is None:
            from .attributes import SeasonalTotal
            if not isinstance(self.attribute, SeasonalTotal):
                raise ValueError("probabilities=None uses the main rainfall-total forecast and requires SeasonalTotal")
        if self.name is None:
            self.name = getattr(self.attribute, "name", "attribute")
        if not (np.isfinite(self.tolerance) and self.tolerance > 0):
            raise ValueError("tolerance must be positive")


def classify_fixed(values, thresholds):
    """Classes with fixed limits: values (n, site), thresholds (2, site)."""
    thresholds = np.asarray(thresholds, float)
    if thresholds.ndim < 1 or thresholds.shape[0] != 2:
        raise ValueError("thresholds must have a leading dimension of length two")
    lo, hi = thresholds
    if np.any(np.isfinite(lo) & np.isfinite(hi) & (hi < lo)):
        raise ValueError("thresholds must be in ascending order")
    v = np.asarray(values, float)
    c = np.where(v <= lo, 0, np.where(v <= hi, 1, 2)).astype(np.int8)
    return np.where(np.isfinite(v) & np.isfinite(lo) & np.isfinite(hi), c, -1).astype(np.int8)


def thresholds_for(constraint, target=None, n_sites=None):
    """(2, site) array of a constraint's fixed thresholds, or None."""
    thr = constraint.thresholds
    if thr is None:
        return None
    if isinstance(thr, xr.DataArray):
        extra = [d for d in thr.dims if d not in ("Y", "X")]
        if len(extra) != 1 or thr.sizes[extra[0]] != 2 or not {"Y", "X"}.issubset(thr.dims):
            raise ValueError("threshold DataArray must have dimensions (threshold=2, Y, X)")
        dim = extra[0]
        if target is not None and not (np.array_equal(thr["Y"].values, target["Y"].values)
                                       and np.array_equal(thr["X"].values, target["X"].values)):
            thr = thr.sortby("Y").sortby("X").interp(Y=target["Y"], X=target["X"], method="nearest")
        values = thr.transpose(dim, "Y", "X").values.reshape(2, -1)
        if np.any(np.isfinite(values).all(0) & (values[1] < values[0])):
            raise ValueError("thresholds must be in ascending order")
        return values
    lo, hi = (float(v) for v in thr)
    if not np.isfinite([lo, hi]).all() or hi < lo:
        raise ValueError("thresholds must be (low, high)")
    return np.array([[lo], [hi]]) * np.ones((1, n_sites or 1))


def _features(categories):
    """(year, site) classes -> indicators of class 0 and 2: (year, site, 2)."""
    return np.stack([(categories == 0), (categories == 2)], axis=-1).astype(float)


def solve_weights(features, targets, prior, tau, method="mre", max_iter=100, tol=1e-9):
    """Batched dual Newton for the relaxed MRE / Croley problem.

    features (year, site, K), targets (site, K), prior (year, site) >= 0 with
    zero for excluded years, tau (K,). Returns weights (year, site) and info.
    """
    g = np.asarray(features, float)
    t = np.asarray(targets, float)
    w0 = np.asarray(prior, float)
    if g.ndim != 3 or min(g.shape) == 0:
        raise ValueError("features must be nonempty (year, site, constraint)")
    ny, ns, K = g.shape
    if t.shape != (ns, K) or w0.shape != (ny, ns):
        raise ValueError("targets and prior must match the feature site/constraint and year/site shapes")
    if not np.isfinite(g).all() or not np.isfinite(t).all():
        raise ValueError("features and targets must be finite")
    if not np.isfinite(w0).all() or (w0 < 0).any():
        raise ValueError("prior must be finite and nonnegative")
    tau = np.broadcast_to(np.asarray(tau, float), (K,))
    if not np.isfinite(tau).all() or (tau <= 0).any():
        raise ValueError("tau must contain finite positive tolerances")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("max_iter and tol must be positive")
    tau2 = tau ** 2
    active = w0.sum(axis=0) > 0
    w0 = np.divide(w0, w0.sum(axis=0, keepdims=True), out=np.zeros_like(w0), where=active[None])
    lam = np.zeros((ns, K))

    if method == "mre":
        logw0 = np.where(w0 > 0, np.log(np.where(w0 > 0, w0, 1)), -np.inf)

        def evaluate(l):
            a = logw0 + np.einsum("ysk,sk->ys", g, l)
            lz = logsumexp(a, axis=0)
            # Inactive sites have log Z = -inf and must never form -inf - -inf.
            w = np.exp(a - np.where(active, lz, 0.)[None])
            w = np.where(np.isfinite(w), w, 0.)
            obj = lz - np.einsum("sk,sk->s", l, t) + 0.5 * np.einsum("k,sk->s", tau2, l ** 2)
            m = np.einsum("ys,ysk->sk", w, g)
            grad = m - t + tau2 * l
            cov = np.einsum("ys,ysk,ysj->skj", w, g, g) - m[:, :, None] * m[:, None, :]
            hess = cov + np.eye(K) * tau2
            return obj, grad, hess, w
    elif method == "croley":
        n = float(ny)
        ones = np.ones((ny, ns, 1))
        G = np.concatenate([g, ones], axis=-1)          # last row: sum-to-one, exact
        t = np.concatenate([t, np.ones((ns, 1))], axis=-1)
        tau2 = np.r_[tau2, 1e-12]
        lam = np.zeros((ns, K + 1))
        eligible = (w0 > 0) | (np.asarray(prior) > 0)

        def evaluate(l):
            w = np.maximum(0., w0 + np.einsum("ysk,sk->ys", G, l) / n) * eligible
            Aw = np.einsum("ys,ysk->sk", w, G)
            obj = (0.5 * n * ((w - w0) ** 2).sum(0) - 0.5 * np.einsum("k,sk->s", tau2, l ** 2)
                   - np.einsum("sk,sk->s", l, Aw - t))
            grad = t - tau2 * l - Aw
            act = (w > 0).astype(float)
            hess = -(np.einsum("ys,ysk,ysj->skj", act, G, G) / n + np.eye(K + 1) * tau2)
            return -obj, -grad, -hess, w          # turn into a minimisation
    else:
        raise ValueError("method must be 'mre' or 'croley'")

    obj, grad, hess, w = evaluate(lam)
    converged = np.zeros(ns, dtype=bool)
    it = 0
    for it in range(1, max_iter + 1):
        converged = (np.abs(grad).max(axis=1) < tol) | ~active
        if converged.all():
            break
        ridge = 1e-12 * np.eye(hess.shape[-1])
        step = -np.linalg.solve(hess + ridge, grad[..., None])[..., 0]
        step[converged] = 0.
        size = np.ones(ns)
        new_lam = lam + step
        new_obj, *_ = evaluate(new_lam)
        # Near the optimum the objective change falls below float precision:
        # there the pure Newton step is taken (quadratic convergence).
        near = np.abs(grad).max(axis=1) < 1e-6
        for _ in range(40):
            bad = ~(new_obj <= obj + 1e-4 * size * np.einsum("sk,sk->s", grad, step)) & ~converged & ~near
            if not bad.any():
                break
            size = np.where(bad, size / 2, size)
            new_lam = lam + size[:, None] * step
            new_obj, *_ = evaluate(new_lam)
        lam = new_lam
        obj, grad, hess, w = evaluate(lam)
    w[:, ~active] = 0.
    # Report convergence of the returned iterate, including the last update.
    converged = (np.abs(grad).max(axis=1) < tol) | ~active
    if method == "croley":
        mass = w.sum(axis=0)
        w = np.divide(w, mass, out=np.zeros_like(w), where=mass > 0)
    gg = g
    achieved = np.einsum("ys,ysk->sk", w, gg)
    info = {"lambda": lam, "achieved": achieved, "iterations": it, "converged": converged,
            "effective_years": np.divide(1., (w ** 2).sum(0), out=np.zeros(ns), where=active),
            "kl_from_prior": np.where(active, (w * np.log(np.where(w > 0, w, 1) / np.where(w0 > 0, w0, 1))).sum(0), np.nan)}
    return w, info


def constrained_year_weights(attribute_values: dict, probabilities: dict, years, climatology=(1991, 2020),
                             prior=None, tolerances=None, method="mre", tercile_method="empirical",
                             miss_tolerance=0.05, min_effective_years=5.0, thresholds=None):
    """Year weights (year, site) satisfying several tercile forecasts at once.

    attribute_values : {name: (year, site)} historical season attributes
    probabilities    : {name: (3, site)} forecast probabilities, low -> high
    prior            : (year,) or (year, site) nonnegative prior weights, or None
    tolerances       : {name: tau}; default 0.01 for every constraint
    thresholds       : {name: (2, site)} fixed class limits replacing terciles

    Returns ``(weights, flags, info)``. Flags extend :func:`year_weights`:
    2 invalid forecast, 4 no history, 8 a constraint missed by more than
    ``miss_tolerance``, 16 fewer than ``min_effective_years`` effective years.
    ``info`` holds per-constraint categories, thresholds, target and achieved
    (3, site) probabilities, effective years and KL divergence from the prior.
    """
    names = list(attribute_values)
    if not names or set(names) != set(probabilities):
        raise ValueError("attribute_values and probabilities need the same constraint names")
    years = np.asarray(years)
    first = np.asarray(attribute_values[names[0]], float)
    if first.ndim != 2 or min(first.shape) == 0 or len(years) != first.shape[0]:
        raise ValueError("attribute values must be nonempty (year, site), with one year per row")
    ny, ns = first.shape
    tolerances = {k: 0.01 for k in names} | dict(tolerances or {})
    fixed_limits = dict(thresholds or {})
    cats, limits, normalized, feats, targets, taus = {}, {}, {}, [], [], []
    valid_year = np.ones((ny, ns), dtype=bool)
    flags = np.zeros(ns, dtype=np.uint8)
    for name in names:
        values = np.asarray(attribute_values[name], float)
        if values.shape != (ny, ns):
            raise ValueError(f"{name}: attribute values must be (year, site) = {(ny, ns)}")
        fixed = fixed_limits.get(name)
        if fixed is not None:
            thr = np.broadcast_to(np.asarray(fixed, float), (2, ns))
            c = classify_fixed(values, thr)
        else:
            c, thr = classify_seasons(values, years, climatology=climatology, method=tercile_method)
        cats[name], limits[name] = c, thr
        valid_year &= c >= 0
        p = np.asarray(probabilities[name], float)
        if p.shape != (3, ns):
            raise ValueError(f"{name}: probabilities must be (3, site)")
        total = p.sum(0)
        bad = ~np.isfinite(p).all(0) | (p < 0).any(0) | (np.abs(total - 1) > 0.02001)
        flags[bad] |= FLAG_INVALID_PROBABILITY
        p = np.divide(p, total, out=np.full_like(p, 1 / 3), where=~bad)
        normalized[name] = np.where(bad[None], np.nan, p)
        feats.append(_features(c))
        targets.append(np.stack([p[0], p[2]], axis=-1))
        taus += [tolerances[name]] * 2
    if prior is None:
        w0 = valid_year.astype(float)
    else:
        pr = np.asarray(prior, float)
        pr = np.broadcast_to(pr[:, None] if pr.ndim == 1 else pr, (ny, ns))
        if not np.isfinite(pr).all() or (pr < 0).any():
            raise ValueError("prior must be finite and nonnegative")
        w0 = pr * valid_year
    invalid = (flags & FLAG_INVALID_PROBABILITY).astype(bool)
    w0[:, invalid] = 0.
    flags[(w0.sum(0) == 0) & ~invalid] |= FLAG_NO_HISTORY
    weights, info = solve_weights(np.concatenate(feats, -1), np.concatenate(targets, -1), w0,
                                  np.asarray(taus), method=method)
    achieved, target = {}, {}
    for name in names:
        c = cats[name]
        achieved[name] = np.stack([(weights * (c == k)).sum(0) for k in range(3)])
        target[name] = normalized[name]
        miss = np.max(np.where(np.isfinite(target[name]), np.abs(achieved[name] - target[name]), 0.), axis=0) > miss_tolerance
        flags[miss & ~invalid] |= FLAG_CONSTRAINT_MISSED
    low = (info["effective_years"] < min_effective_years) & ~invalid
    flags[low] |= FLAG_LOW_EFFECTIVE_SIZE
    info.update(categories=cats, thresholds=limits, achieved_probability=achieved,
                target_probability=target, names=names, method=method)
    return weights, flags, info


def compute_constraint_attributes(constraints, prcp: xr.DataArray, years):
    """Evaluate every constraint's attribute on daily PRCP -> {name: (year, site)}."""
    out = {}
    for con in constraints:
        values = con.attribute.compute(prcp, years)
        out[con.name] = values.transpose("season_year", ...).values.reshape(len(years), -1)
    return out


__all__ = ["SeasonalConstraint", "constrained_year_weights", "solve_weights", "relabel_probabilities",
           "compute_constraint_attributes", "FLAG_CONSTRAINT_MISSED", "FLAG_LOW_EFFECTIVE_SIZE"]
