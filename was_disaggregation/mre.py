"""Multi-constraint historical year weights: minimum relative entropy and Croley.

Given K season attributes (total, onset, dry spells, ...) each with a tercile
forecast, find year weights w (per site) that

    minimise   D(w, w0) + sum_k sum_c (P_w[class_k = c] - p_kc)^2 / (2 tau_k^2)
    subject to w >= 0, sum w = 1,

where w0 is a prior (uniform, or e.g. ENSO rank weights) and

    D = KL(w || w0)                      method='mre'   (Weijs & van de Giesen 2013)
    D = (n/2) sum (w - w0)^2             method='croley' (Croley 1996, 2000)

This is the legacy soft mode. Exact mode instead minimizes D(w,w0) subject to
the feasible equalities P_w[class_k = c] = p_kc. The tolerance tau_k is a
penalty scale rather than a strict bound on probability error: with
compatible forecasts every constraint is met to within ~tau_k^2 * lambda; with
conflicting forecasts (e.g. wet total AND late onset when history says late
onsets are dry) the solution trades the constraints off in proportion to 1/tau_k^2
instead of failing. All three categories enter the soft loss so exchanging
category labels cannot change the objective. Their exact equalities are
linearly dependent and redundant rows are removed only after feasibility is
checked.

MRE solution: w_y = w0_y exp(lambda . g_y) / Z with g_y the class indicators.
The soft objectives are solved in the dual by batched damped Newton. Exact
mode first tests support feasibility by linear programming, removes redundant
equalities by QR, then solves a constrained primal program with SLSQP per site.

Scientific scope
----------------
The exact MRE problem uses the published minimum-relative-entropy principle.
The generalized Euclidean prior-distance objective is Croley-inspired; it
reduces to the usual uniform-prior quadratic adjustment up to scaling when
the prior is uniform. Symmetric three-category soft penalties, nonuniform
priors, support checks and regional fraction constraints are explicit package
extensions. Exact donor constraints do not establish exact generated weather
attributes after a parametric fit or daily resampling.

References
----------
* Stefan V. Weijs and Nick van de Giesen (2013), "An Information-Theoretical
  Perspective on Weighted Ensemble Forecasts".
  https://doi.org/10.1016/j.jhydrol.2013.06.033
  Core: minimize KL divergence from prior weights under new forecast constraints.
* Thomas E. Croley II (1996), "Using NOAA's New Climate Outlooks in Operational
  Hydrology". https://doi.org/10.1061/(ASCE)1084-0699(1996)1:3(93)
  Background: constrained adjustment of historical scenario probabilities;
  the general-prior Euclidean and soft formulations here are declared variants.
* Thomas E. Croley II (2000), "Using Meteorology Probability Forecasts in
  Operational Hydrology". https://doi.org/10.1061/9780784404591
  Background: ensemble probabilities reflecting meteorological event forecasts.
* William M. Briggs and Daniel S. Wilks (1996), "Extension of the Climate
  Prediction Center Long-Lead Temperature and Precipitation Outlooks to General
  Weather Statistics". https://doi.org/10.1175/1520-0442(1996)009<3496:EOTCPC>2.0.CO;2
  Core special case: a single categorical forecast gives p(category)/n(category).

Scientific attributions are separate from software authorship recorded in
package metadata and copyright notices. Published application experiments are
not reproduced by solving these finite optimization problems alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warnings
import xarray as xr
from scipy.linalg import qr
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from scipy.special import logsumexp, xlogy

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
    """All three indicators; redundancy is harmless and keeps soft loss symmetric."""
    return np.stack([(categories == c) for c in range(3)], axis=-1).astype(float)


def _solve_exact_site(g, target, prior, method, max_iter, tol, site):
    """Constrained primal optimization on the positive support of one prior."""
    supported = prior > 0
    A = np.vstack((np.ones(supported.sum()), g[supported].T))
    b = np.r_[1., target]
    feasible = linprog(np.zeros(supported.sum()), A_eq=A, b_eq=b,
                       bounds=(0, None), method="highs")
    if not feasible.success:
        raise ValueError(f"site {site}: exact forecast constraints are infeasible on the prior support")
    # Duplicate or linearly dependent class indicators are common. SLSQP
    # requires an independent equality system, after feasibility has been checked.
    _, R, pivot = qr(A.T, mode="economic", pivoting=True)
    rank = int(np.sum(np.abs(np.diag(R)) > 1e-11 * max(A.shape)
                      * (np.abs(R).max() if R.size else 1.)))
    rows = np.sort(pivot[:rank])
    equalities = LinearConstraint(A[rows], b[rows], b[rows])
    w0 = prior[supported] / prior[supported].sum()
    if method == "mre":
        objective = lambda w: float(np.sum(xlogy(w, w / w0)))
        gradient = lambda w: np.log(np.maximum(w, 1e-300) / w0) + 1.
    else:
        n = float(len(prior))
        objective = lambda w: float(n * np.sum((w - w0) ** 2) / 2)
        gradient = lambda w: n * (w - w0)
    result = minimize(objective, feasible.x, jac=gradient, method="SLSQP",
                      constraints=[equalities], bounds=Bounds(0, np.inf),
                      options={"maxiter": max_iter, "ftol": min(tol, 1e-12)})
    residual = np.max(np.abs(A @ result.x - b))
    if not result.success or residual > max(1e-8, 10 * tol):
        raise RuntimeError(f"site {site}: exact {method} optimization did not converge: {result.message}")
    weights = np.zeros(len(prior), float)
    weights[supported] = result.x
    return weights, result.nit


def solve_weights(features, targets, prior, tau, method="mre", max_iter=100, tol=1e-9,
                  constraint_mode="soft"):
    """Optimize MRE / Croley weights with soft or exact forecast constraints.

    Scientific sources: Weijs and van de Giesen (2013), DOI
    10.1016/j.jhydrol.2013.06.033, and Croley (1996, 2000), listed in the module
    References. ``mre`` uses KL; ``croley`` is the package's generalized
    Euclidean prior-distance variant. Symmetric soft penalties are extensions.

    features (year, site, K), targets (site, K), prior (year, site) >= 0 with
    zero for excluded years, tau (K,). Returns weights (year, site) and info.
    ``constraint_mode='soft'`` retains the 0.8.0 quadratic penalty with positive
    ``tau``. ``'exact'`` imposes all target equalities within numerical tolerance
    on the support of the prior and raises ValueError for infeasible sites. The
    exact option solves a separate constrained program per site, so it is slower.
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
    if method not in {"mre", "croley"}:
        raise ValueError("method must be 'mre' or 'croley'")
    if constraint_mode not in {"soft", "exact"}:
        raise ValueError("constraint_mode must be 'soft' or 'exact'")
    tau2 = tau ** 2
    active = w0.sum(axis=0) > 0
    w0 = np.divide(w0, w0.sum(axis=0, keepdims=True), out=np.zeros_like(w0), where=active[None])
    if constraint_mode == "exact":
        weights = np.zeros_like(w0)
        iterations = np.zeros(ns, int)
        for site in np.flatnonzero(active):
            weights[:, site], iterations[site] = _solve_exact_site(
                g[:, site], t[site], w0[:, site], method, max_iter, tol, site)
        achieved = np.einsum("ys,ysk->sk", weights, g)
        info = {"lambda": None, "achieved": achieved,
                "iterations": iterations, "converged": np.ones(ns, dtype=bool),
                "effective_years": np.divide(1., (weights ** 2).sum(0), out=np.zeros(ns), where=active),
                "kl_from_prior": np.where(active, (
                    weights * np.log(np.where(weights > 0, weights, 1) /
                                     np.where(w0 > 0, w0, 1))).sum(0), np.nan),
                "constraint_mode": constraint_mode}
        info["active"] = active
        info["max_constraint_error"] = np.max(np.abs(achieved - t), axis=1)
        return weights, info
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
    if np.any(active & ~converged):
        warnings.warn(f"{method} solver did not converge at "
                      f"{int(np.sum(active & ~converged))} active site(s); "
                      "inspect info['converged'] before using the weights.",
                      RuntimeWarning, stacklevel=2)
    if method == "croley":
        mass = w.sum(axis=0)
        w = np.divide(w, mass, out=np.zeros_like(w), where=mass > 0)
    gg = g
    achieved = np.einsum("ys,ysk->sk", w, gg)
    info = {"lambda": lam, "achieved": achieved, "iterations": it, "converged": converged,
            "effective_years": np.divide(1., (w ** 2).sum(0), out=np.zeros(ns), where=active),
            "kl_from_prior": np.where(active, (w * np.log(np.where(w > 0, w, 1) / np.where(w0 > 0, w0, 1))).sum(0), np.nan),
            "constraint_mode": constraint_mode}
    info["active"] = active
    info["max_constraint_error"] = np.max(np.abs(achieved - np.asarray(targets)), axis=1)
    return w, info


def constrained_year_weights(attribute_values: dict, probabilities: dict, years, climatology=(1991, 2020),
                             prior=None, tolerances=None, method="mre", tercile_method="empirical",
                             miss_tolerance=0.05, min_effective_years=5.0, thresholds=None,
                             constraint_mode="soft"):
    """Year weights (year, site) targeting several tercile forecasts at once.

    Uses the MRE principle of Weijs and van de Giesen (2013), or a declared
    Croley-inspired quadratic variant; complete references are in the module
    docstring. Exact mode satisfies feasible donor targets to numerical
    tolerance; soft mode trades off errors through the package penalty.

    attribute_values : {name: (year, site)} historical season attributes
    probabilities    : {name: (3, site)} forecast probabilities, low -> high
    prior            : (year,) or (year, site) nonnegative prior weights, or None
    tolerances       : {name: tau}; default 0.01 for every constraint
    thresholds       : {name: (2, site)} fixed class limits replacing terciles
    constraint_mode  : 'soft' (legacy penalty) or 'exact' (feasible equalities)

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
            c, thr = classify_seasons(values, years, climatology=climatology,
                                     method=tercile_method,
                                     allow_negative=tercile_method == "empirical")
        cats[name], limits[name] = c, thr
        valid_year &= c >= 0
        p = np.asarray(probabilities[name], float)
        if p.shape != (3, ns):
            raise ValueError(f"{name}: probabilities must be (3, site)")
        total = p.sum(0)
        bad = (~np.isfinite(p).all(0) | (p < 0).any(0) |
               ~((np.abs(total - 1) <= 0.020000001) |
                 (np.abs(total - 100) <= 2.0000001)))
        flags[bad] |= FLAG_INVALID_PROBABILITY
        p = np.divide(p, total, out=np.full_like(p, 1 / 3), where=~bad)
        normalized[name] = np.where(bad[None], np.nan, p)
        feats.append(_features(c))
        targets.append(p.T)
        taus += [tolerances[name]] * 3
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
                                  np.asarray(taus), method=method, constraint_mode=constraint_mode)
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
