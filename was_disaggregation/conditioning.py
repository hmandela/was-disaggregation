"""Climatological categories and forecast-conditioned historical year weights.

Scientific scope
----------------
The Briggs--Wilks category-mass identity and Stedinger--Kim density-ratio
identity are implemented directly. Empirical/Gamma/normal category estimators,
empty-category policies, normal-score forecast densities, and optional
within-category recalibration are explicit package choices. Three forecast
probabilities do not identify a unique continuous forecast distribution.
Yates/Clark rank selection is a component, not a reproduction of their complete
regional kNN generator, climate scenarios, or RPSS optimization experiments.
Historical category masses do not guarantee simulated seasonal category masses.

References
----------
* William M. Briggs and Daniel S. Wilks (1996), "Extension of the Climate
  Prediction Center Long-Lead Temperature and Precipitation Outlooks to General
  Weather Statistics". https://doi.org/10.1175/1520-0442(1996)009<3496:EOTCPC>2.0.CO;2
  Core: historical-year weights conditional on seasonal categories.
* Daniel S. Wilks (2002), "Realizations of Daily Weather in Forecast Seasonal
  Climate". https://doi.org/10.1175/1525-7541(2002)003<0195:RODWIF>2.0.CO;2
  Core: Gamma precipitation and normal temperature climatological categories.
* Jery R. Stedinger and Young-Oh Kim (2010), "Probabilities for Ensemble
  Forecasts Reflecting Climate Information".
  https://doi.org/10.1016/j.jhydrol.2010.06.038
  Core: density-ratio weights; forecast-density construction and category
  recalibration here are declared variants.
* David Yates, Subhrendu Gangopadhyay, Balaji Rajagopalan and Kenneth Strzepek
  (2003), "A Technique for Generating Regional Climate Scenarios Using a
  Nearest-Neighbor Algorithm". https://doi.org/10.1029/2002WR001769
  Background: preferential rank selection, not the complete scenario generator.
* Martyn P. Clark, Subhrendu Gangopadhyay, David Brandon, Kevin Werner,
  Lauren E. Hay, Balaji Rajagopalan and David Yates (2004), "A Resampling
  Procedure for Generating Conditioned Daily Weather Sequences".
  https://doi.org/10.1029/2003WR002747
  Core component: Eq. (9) rank-selection probabilities integrated analytically.
* Rob J. Hyndman and Yanan Fan (1996), "Sample Quantiles in Statistical
  Packages". https://doi.org/10.1080/00031305.1996.10473566
  Numerical convention: type 5 (Hazen) and type 7 (linear) quantiles.
* Mandela Coovi Mahuwetin Houngnibo, Abdou Ali, Alhassane Agali, Moussa Waongo,
  Agnide Emmanuel Lawin and Jean-Martial Cohard (2023), "Stochastic
  Disaggregation of Seasonal Precipitation Forecasts of the West African
  Regional Climate Outlook Forum". https://doi.org/10.1002/joc.8161
  Protocol component: empirical Hazen category thresholds.

These are scientific-method attributions; software authorship is recorded
separately in the package metadata and copyright notices.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy import stats
from scipy.special import logsumexp

FLAG_OK = 0
FLAG_EMPTY_TERCILE = 1
FLAG_INVALID_PROBABILITY = 2
FLAG_NO_HISTORY = 4


def classify_seasons(totals, years, climatology=(1991, 2020), method="empirical",
                     quantile_method="linear", *, allow_negative=False):
    """Classify complete seasonal rainfall totals using fixed reference years.

    ``totals`` is year,site and must already be computed with ``skipna=False``.
    PB: x <= q1; PN: q1 < x <= q2; PA: x > q2. Missing/negative totals
    or cells with fewer than three reference years receive category -1.
    Gamma uses a fitted zero-inflated Gamma; degenerate/insufficient positive
    samples fall back to empirical thresholds. Ties are never randomized.
    ``quantile_method='hazen'`` selects the Hyndman–Fan type 5 sample
    quantiles used by Houngnibo et al. (2023); the default ``'linear'``
    retains NumPy's type 7 quantiles from the 0.8.0 release. The choice
    also applies to empirical fallbacks from a Gamma fit. Use
    :func:`classify_values` for signed seasonal temperature statistics.
    """
    values = np.asarray(totals, dtype=float)
    years = np.asarray(years)
    if values.ndim != 2 or years.ndim != 1 or values.shape[0] != years.size:
        raise ValueError("totals must be (year,site), years must match the first dimension.")
    if len(np.unique(years)) != years.size:
        raise ValueError("Season years must be unique.")
    if method not in {"empirical", "gamma"}:
        raise ValueError("method must be 'empirical' or 'gamma'.")
    if quantile_method not in {"linear", "hazen"}:
        raise ValueError("quantile_method must be 'linear' or 'hazen'.")
    if not isinstance(allow_negative, (bool, np.bool_)):
        raise ValueError("allow_negative must be a bool")
    if allow_negative and method == "gamma":
        raise ValueError("gamma requires nonnegative seasonal totals")
    if len(climatology) != 2 or climatology[0] > climatology[1]:
        raise ValueError("climatology must be (first_year,last_year), inclusive.")
    reference = (years >= climatology[0]) & (years <= climatology[1])
    if not reference.any():
        raise ValueError("No historical season years overlap the requested climatology.")
    valid = np.isfinite(values) & (allow_negative | (values >= 0))
    thresholds = np.full((2, values.shape[1]), np.nan)
    categories = np.full(values.shape, -1, dtype=np.int8)
    for site in range(values.shape[1]):
        sample = values[reference & valid[:, site], site]
        if sample.size < 3:
            continue
        quantiles = np.quantile(sample, [1. / 3., 2. / 3.], method=quantile_method)
        if method == "gamma":
            positive = sample[sample > 0]
            p_zero = 1. - positive.size / sample.size
            if positive.size >= 3 and np.std(positive) > 1.e-12 * max(np.mean(positive), 1.):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        shape, _, scale = stats.gamma.fit(positive, floc=0)
                    conditional_q = (np.array([1. / 3., 2. / 3.]) - p_zero) / (1. - p_zero)
                    gamma_q = np.zeros(2)
                    above_zero = conditional_q > 0
                    gamma_q[above_zero] = stats.gamma.ppf(conditional_q[above_zero], shape, scale=scale)
                    if np.isfinite(gamma_q).all():
                        quantiles = gamma_q
                except (ValueError, RuntimeError, FloatingPointError, RuntimeWarning):
                    pass  # The empirical fallback is explicit in this method's contract.
        thresholds[:, site] = quantiles
        sample_valid = valid[:, site]
        categories[sample_valid, site] = np.where(values[sample_valid, site] <= quantiles[0], 0,
            np.where(values[sample_valid, site] <= quantiles[1], 1, 2))
    return categories, thresholds


def classify_values(values, years, climatology=(1991, 2020), quantile_method="linear",
                    method="empirical"):
    """Terciles of a signed seasonal statistic such as mean TMAX/TMIN.

    With ``quantile_method='hazen'``, use the same type 5 plotting positions
    as Houngnibo et al. (2023). Unlike :func:`classify_seasons`, finite
    negative values are legitimate observations (e.g. Celsius in winter).
    With ``method='normal'``, use Wilks's (2002) Gaussian seasonal-temperature
    thresholds ``mean + Phi^-1(1/3, 2/3) * sample_std``, where sample_std
    uses ``ddof=1`` on the fixed reference years. ``quantile_method`` has no
    effect on the Gaussian thresholds.
    Returns ``(categories, thresholds)`` with the same shape convention.
    """
    if method not in {"empirical", "normal"}:
        raise ValueError("method must be 'empirical' or 'normal'")
    categories, thresholds = classify_seasons(
        values, years, climatology=climatology, method="empirical",
        quantile_method=quantile_method, allow_negative=True)
    if method == "empirical":
        return categories, thresholds
    values = np.asarray(values, dtype=float)
    years = np.asarray(years)
    reference = (years >= climatology[0]) & (years <= climatology[1])
    z_terciles = stats.norm.ppf([1. / 3., 2. / 3.])
    for site in range(values.shape[1]):
        column = values[:, site]
        valid = np.isfinite(column)
        sample = column[reference & valid]
        if sample.size < 3:
            continue
        q = sample.mean() + z_terciles * sample.std(ddof=1)
        thresholds[:, site] = q
        categories[valid, site] = np.where(column[valid] <= q[0], 0,
                                            np.where(column[valid] <= q[1], 1, 2))
    return categories, thresholds


def year_weights(categories, probabilities, empty_policy="climatology"):
    """Return p(category)/n(category) per historical year and site plus flags.

    Briggs and Wilks (1996); Wilks (2002), cited in the module References.
    Exact historical category masses require donors in every targeted class.
    Empty-category policies below are package support-handling extensions.

    Flags are uint8 bits: 1=empty positive-probability category, 2=invalid
    forecast, 4=no complete history. Invalid forecasts/history receive zero
    weights. Empty-category default is uniform complete historical years;
    ``renormalize`` reallocates only across supported forecast categories;
    ``raise`` rejects unsupported target mass.
    """
    cats = np.asarray(categories)
    probs = np.asarray(probabilities, dtype=float)
    if cats.ndim != 2 or probs.shape != (3, cats.shape[1]):
        raise ValueError("categories must be (year,site), probabilities must be (3,site).")
    if not np.isin(cats, [-1, 0, 1, 2]).all():
        raise ValueError("Categories must be -1 (missing), 0, 1, or 2.")
    if empty_policy not in {"climatology", "renormalize", "raise"}:
        raise ValueError("empty_policy must be 'climatology', 'renormalize', or 'raise'.")
    weights = np.zeros(cats.shape, dtype=float)
    flags = np.zeros(cats.shape[1], dtype=np.uint8)
    for site in range(cats.shape[1]):
        valid = cats[:, site] >= 0
        if not valid.any():
            flags[site] |= FLAG_NO_HISTORY
        # Empty-category renormalization must not rewrite the caller's
        # forecast array, which may be reused by other fitted methods.
        p = probs[:, site].copy()
        total = p.sum()
        if (not np.isfinite(p).all() or (p < 0).any()
                or not (abs(total - 1.) <= .020000001 or abs(total - 100.) <= 2.0000001)):
            flags[site] |= FLAG_INVALID_PROBABILITY
            continue
        if not valid.any():
            continue
        p = p / total
        counts = np.bincount(cats[valid, site].astype(int), minlength=3)
        empty_target = (counts == 0) & (p > 0)
        if empty_target.any():
            flags[site] |= FLAG_EMPTY_TERCILE
            if empty_policy == "raise":
                raise ValueError(f"Site {site} has positive forecast mass in empty categories {np.flatnonzero(empty_target).tolist()}.")
            if empty_policy == "climatology":
                weights[valid, site] = 1. / valid.sum()
                continue
            p[counts == 0] = 0.
            if p.sum() <= 0:
                # No supported forecast mass exists; explicit climatology fallback
                # is preferable to a division by zero and retains the empty flag.
                weights[valid, site] = 1. / valid.sum()
                continue
            p /= p.sum()
        for category in range(3):
            if counts[category]:
                weights[cats[:, site] == category, site] = p[category] / counts[category]
    return weights, flags


def pdf_ratio_weights(values, log_pdf_forecast, log_pdf_climatology):
    """Stable Stedinger–Kim importance weights along the historical axis (0).

    Densities may be callables evaluated at ``values`` or already evaluated
    arrays broadcastable to its shape. NaN values/densities have zero weight.
    A forecast density cannot put positive mass outside climatological support:
    such a supplied density ratio raises ValueError. Columns with no supported
    positive forecast density receive zero weights and should be excluded.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        raise ValueError("values must include a historical sample axis.")
    f = log_pdf_forecast(values) if callable(log_pdf_forecast) else log_pdf_forecast
    c = log_pdf_climatology(values) if callable(log_pdf_climatology) else log_pdf_climatology
    f = np.broadcast_to(np.asarray(f, dtype=float), values.shape)
    c = np.broadcast_to(np.asarray(c, dtype=float), values.shape)
    finite_values = np.isfinite(values)
    if (finite_values & np.isneginf(c) & np.isfinite(f)).any():
        raise ValueError("Forecast density is positive outside climatological support; pdf ratio is undefined.")
    valid = finite_values & np.isfinite(c) & ~np.isnan(f) & ~np.isposinf(f)
    log_ratio = np.full(values.shape, -np.inf)
    with np.errstate(over="ignore", invalid="ignore"):
        np.subtract(f, c, out=log_ratio, where=valid)
    # Overflow from finite log densities is dominated by the +inf terms.
    dominant = np.isposinf(log_ratio)
    counts = dominant.sum(axis=0, keepdims=True)
    normalizer = logsumexp(log_ratio, axis=0, keepdims=True)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        ordinary = np.exp(log_ratio - normalizer)
        weights = np.where(counts > 0, dominant / np.maximum(counts, 1), ordinary)
    return np.where(np.isfinite(weights), weights, 0.)


# ---------------------------------------------------------------------------
# v0.2.0 additions: normal scores, tercile-matched pdf-ratio weights, class
# weights for mixture conditioning, and Yates/Clark rank-selection masses.
# ---------------------------------------------------------------------------
_Z33 = float(stats.norm.ppf(1. / 3.))


def normal_scores(totals, years, climatology=(1991, 2020), method="empirical"):
    """Normal score z = Phi^-1(F0(x)) of each historical seasonal total.

    F0 is the climatological CDF of the reference years: mid-rank plotting
    positions ``(n_below + n_equal/2 + 1/2) / (n_ref + 1)`` for ``empirical``
    (ties such as zero totals share one score), or the zero-inflated Gamma of
    :func:`classify_seasons` for ``gamma``. Returns (year, site); NaN where the
    total is missing or fewer than three reference years exist.
    """
    values = np.asarray(totals, dtype=float)
    years = np.asarray(years)
    if values.ndim != 2 or years.ndim != 1 or values.shape[0] != years.size:
        raise ValueError("totals must be (year,site), years must match the first dimension.")
    if len(np.unique(years)) != years.size:
        raise ValueError("Season years must be unique.")
    if method not in {"empirical", "gamma"}:
        raise ValueError("method must be 'empirical' or 'gamma'.")
    if len(climatology) != 2 or climatology[0] > climatology[1]:
        raise ValueError("climatology must be (first_year,last_year), inclusive.")
    reference = (years >= climatology[0]) & (years <= climatology[1])
    if not reference.any():
        raise ValueError("No historical season years overlap the requested climatology.")
    z = np.full(values.shape, np.nan)
    for site in range(values.shape[1]):
        column = values[:, site]
        ok = np.isfinite(column) & (column >= 0)
        sample = np.sort(column[reference & ok])
        if sample.size < 3:
            continue
        x = column[ok]
        F = None
        if method == "gamma":
            positive = sample[sample > 0]
            p_zero = 1. - positive.size / sample.size
            if positive.size >= 3 and np.std(positive) > 1.e-12 * max(np.mean(positive), 1.):
                try:
                    shape, _, scale = stats.gamma.fit(positive, floc=0)
                    F = np.where(x > 0, p_zero + (1 - p_zero) * stats.gamma.cdf(x, shape, scale=scale), p_zero / 2)
                except (ValueError, RuntimeError, FloatingPointError):
                    F = None
        if F is None:
            below = np.searchsorted(sample, x, side="left")
            equal = np.searchsorted(sample, x, side="right") - below
            F = (below + 0.5 * equal + 0.5) / (sample.size + 1.)
        z[ok, site] = stats.norm.ppf(np.clip(F, 1e-6, 1 - 1e-6))
    return z


def tercile_normal_forecast(probabilities, min_normal=0.02):
    """Normal N(mu, sigma) in normal-score space whose terciles equal PB, PN, PA.

    Solves Phi((z33-mu)/sigma) = PB and Phi((z67-mu)/sigma) = 1-PA in closed form:
    sigma = (z67-z33)/(Phi^-1(1-PA)-Phi^-1(PB)), mu = z33 - sigma*Phi^-1(PB).
    This is an explicit *assumption* (Stedinger & Kim 2010 use a smooth forecast
    density; three probabilities do not identify one uniquely). PN is floored at
    ``min_normal`` so that sigma stays finite. Fractions and percentages are
    accepted with the same 2% sum tolerance as :func:`year_weights`. Invalid
    forecasts return NaN parameters. Returns (mu, sigma), each (site,).
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim not in (1, 2) or p.shape[0] != 3:
        raise ValueError("probabilities must have shape (3,) or (3,site).")
    if not np.isfinite(min_normal) or not 0 < min_normal < 1:
        raise ValueError("min_normal must be finite and strictly between 0 and 1.")
    total = p.sum(axis=0)
    valid = (np.isfinite(p).all(axis=0) & (p >= 0).all(axis=0)
             & ((abs(total - 1.) <= .020000001) | (abs(total - 100.) <= 2.0000001)))
    p = np.divide(p, total, out=np.full_like(p, np.nan), where=valid)
    pb, pa = np.clip(p[0], 1e-4, 1 - 1e-4), np.clip(p[2], 1e-4, 1 - 1e-4)
    # Scale both tails together rather than subtracting an equal excess: the
    # latter makes the smaller tail negative for deterministic forecasts.
    tail_scale = np.minimum(1., (1 - min_normal) / (pb + pa))
    pb, pa = pb * tail_scale, pa * tail_scale
    a, b = stats.norm.ppf(pb), stats.norm.ppf(1 - pa)
    sigma = (-2 * _Z33) / (b - a)
    return _Z33 - sigma * a, sigma


def tercile_pdf_ratio_weights(zscores, categories, probabilities, calibrate=True, min_normal=0.02,
                             empty_policy="renormalize"):
    """Stedinger–Kim pdf-ratio year weights from a tercile forecast.

    q_y = f1(z_y)/f0(z_y) with f0 = N(0,1) and f1 = :func:`tercile_normal_forecast`.
    Unlike p/N weights, relative preferences within each category depend on the
    entire forecast/climatology density ratio. Tail weights may increase or
    decrease: for a more dispersed normal forecast they increase with |z|.
    With ``calibrate=True`` (default) the q_y
    are rescaled inside each category so that category masses equal PB, PN, PA
    exactly, i.e. the Briggs–Wilks tercile mass is kept and only the
    within-category shape comes from the pdf ratio. ``empty_policy`` follows
    :func:`year_weights`; a climatology fallback remains uniform over complete
    historical years. Returns (weights, flags) with those flag conventions.
    """
    z = np.asarray(zscores, dtype=float)
    cats = np.asarray(categories)
    probs = np.asarray(probabilities, dtype=float)
    if z.shape != cats.shape:
        raise ValueError("zscores must have the same (year, site) shape as categories")
    if not isinstance(calibrate, (bool, np.bool_)):
        raise ValueError("calibrate must be a bool")
    base, flags = year_weights(cats, probs, empty_policy=empty_policy)
    mu, sigma = tercile_normal_forecast(probs, min_normal)
    with np.errstate(invalid="ignore", over="ignore"):
        log_q = stats.norm.logpdf(z, mu[None], sigma[None]) - stats.norm.logpdf(z)
    log_q = np.where(np.isfinite(z) & (cats >= 0), log_q, -np.inf)
    q = np.exp(log_q - np.max(np.where(np.isfinite(log_q), log_q, -1e300), axis=0, keepdims=True))
    q = np.where(np.isfinite(q), q, 0.)
    weights = np.zeros_like(q)
    if calibrate:
        for c in range(3):
            inside = cats == c
            mass = base * inside
            target = mass.sum(axis=0)
            qc = q * inside
            s = qc.sum(axis=0)
            weights += np.divide(qc * target, s, out=mass.copy(), where=s > 0)
    else:
        weights = q
    total = weights.sum(axis=0)
    weights = np.divide(weights, total, out=base.copy(), where=total > 0)
    if empty_policy == "climatology":
        fallback = (flags & FLAG_EMPTY_TERCILE).astype(bool)
        weights[:, fallback] = base[:, fallback]
    invalid = (flags & FLAG_INVALID_PROBABILITY).astype(bool) | ~np.isfinite(probs).all(axis=0)
    weights[:, invalid] = 0.
    return weights, flags


def class_weights(base_weights, categories, fallback_weights=None):
    """Weights restricted to each tercile class and renormalized: (3, year, site).

    Class c of a mixture uses only years classified c, keeping the relative
    weights of ``base_weights`` (uniform for p/N weights, pdf-ratio shaped for
    Stedinger–Kim). A class without any historical year falls back to
    ``fallback_weights`` (normally the forecast-mean weights) for that site.
    """
    w = np.asarray(base_weights, dtype=float)
    cats = np.asarray(categories)
    fb = w if fallback_weights is None else np.asarray(fallback_weights, dtype=float)
    if w.ndim != 2 or cats.shape != w.shape or fb.shape != w.shape:
        raise ValueError("base, categories and fallback must share (year, site) shape")
    if not np.isin(cats, (-1, 0, 1, 2)).all():
        raise ValueError("categories must contain -1, 0, 1 or 2")
    if any(not np.isfinite(a).all() or np.any(a < 0) for a in (w, fb)):
        raise ValueError("base and fallback weights must be finite and nonnegative")
    # Missing historical seasons never gain support through a fallback.
    fb = np.where(cats >= 0, fb, 0.)
    total_fb = fb.sum(axis=0)
    fb = np.divide(fb, total_fb, out=np.zeros_like(fb), where=total_fb > 0)
    out = np.zeros((3,) + w.shape)
    empty = np.zeros((3, w.shape[1]), dtype=bool)
    for c in range(3):
        inside = (cats == c).astype(float)
        wc = np.where(w > 0, w, 0.) * inside
        s = wc.sum(axis=0)
        # tercile p/N weights are zero in classes with p_c = 0: fall back to
        # equal weights of the class years so every class remains defined.
        count = inside.sum(axis=0)
        eq = np.divide(inside, count, out=np.zeros_like(inside), where=count > 0)
        out[c] = np.where(s > 0, np.divide(wc, s, out=np.zeros_like(wc), where=s > 0), eq)
        empty[c] = count == 0
        out[c][:, empty[c]] = fb[:, empty[c]]
    return out, empty


def yates_rank_probabilities(n, strength=1.0, selection=1.0):
    """Exact mass of rank i for i = INT(N * U**strength / selection) + 1.

    Yates et al. (2003) / Clark et al. (2004a): strength (lambda) > 1 favours
    the most similar years, selection (alpha) >= 1 truncates to the best N/alpha.
    Clark et al. (2004), Eq. (9), DOI 10.1029/2003WR002747; complete authors
    and the Yates source are listed in the module References. Analytic masses
    remove finite donor-bootstrap noise; no kNN climate-scenario generator or
    cross-validated optimization of strength/selection is performed here.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer.")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be finite and positive.")
    if not np.isfinite(selection) or selection < 1:
        raise ValueError("selection must be finite and >= 1.")
    i = np.arange(n + 1, dtype=float)
    edges = np.minimum(selection * i / n, 1.) ** (1. / strength)
    return np.diff(edges)
