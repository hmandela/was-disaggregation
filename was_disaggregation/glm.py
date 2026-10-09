"""Verdin-inspired GLM weather generation with seasonal forecast covariates.

Verdin et al. (2018, J. Hydrol.) condition a GLM weather generator (Furrer &
Katz 2007; Kleiber et al. 2012) on seasonal climate. The seasonal total enters
the daily models as a covariate, and seasonal totals are sampled from the
forecast. Verdin et al. (2019, BayGEN) make the coefficients spatial Gaussian
processes estimated in a Bayesian space-time hierarchy.

In contrast to "weight the years, then refit", the forecast is not a
reweighting of history. Every historical season contributes to the fit, and the
forecast only changes the covariate value at simulation time.

Default package model at each cell (not the exact Verdin covariate model)::

    P(wet_t) = Phi(X_occ,t . b_occ),  X_occ = [1, harmonics(doy), wet_{t-1}, z_S, z_S * wet_{t-1}]
    rain_t - thr | wet ~ Gamma(k, mu_t / k),  log mu_t = X_amt,t . b_amt,  X_amt = [1, harmonics, wet_{t-1}, z_S]
    g(x_t) = X_c,t . b_c + sd_{wet/dry} * e_t,   X_c = [1, harmonics, wet_t, g(x_{t-1}), z_S]

``z_S`` is the normal score Phi^-1(F0(S)) of the season's total S, where F0 is
the reference-period climatology. Using the normal score instead of the raw
total makes the covariate scale-free. This is a modelling choice, not the raw
areal seasonal total of Verdin et al. (2018). An optional ``domain_three``
mode uses a domain-mean rain total and domain-mean TMIN/TMAX seasonal anomalies
as separate covariates; its temperature models also use both lagged
temperatures and an interannual trend. It remains an adaptation of the paper.
``paper_protocol='verdin_2018'`` instead uses an affine standardized physical
domain rain total (equivalent to a raw-total linear predictor), one annual
harmonic, and no lagged occurrence in the amount predictor. Raw Gamma amounts
are conditioned to exceed 0.1 mm to respect the simulated wet-day support.
That truncation is a declared extension. This is a seasonal submodel, not the
authors' complete multi-season experiment. Direct ``seasonal_total`` and
``domain_probabilities`` inputs avoid treating the area average of local
category probabilities as an exact forecast of the domain-total categories.

Spatial dependence (Kleiber et al. 2012): occurrence is wet when a latent
spatial Gaussian field W_t < X_occ b_occ. Amounts are Gamma quantiles of
Phi(V_t) with V_t a second field. The other variables have their own fields,
mixed across variables by the per-cell residual correlation. The seasonal
covariate of each member is drawn from a further field, so it stays coherent
in space.

Bayesian mode (``bayesian=True``) is an empirical-Bayes Laplace approximation
of BayGEN, not full MCMC. Each coefficient is given an independent spatial GP
prior, m + GP(0, tau^2 exp(-d/rho)). The per-cell GLM estimates, with their
sampling variances (inverse Fisher information), are the data. The
hyperparameters (m, tau^2, rho) maximise the marginal likelihood. The
posterior (i) smooths noisy cells, (ii) propagates parameter uncertainty (each
member draws its own coefficient field) and (iii) predicts coefficients at
cells without data (``generate(target=grid)``).

References
----------
* Andrew Verdin, Balaji Rajagopalan, William Kleiber, Guillermo Podesta and
  Federico Bert (2018), "A Conditional Stochastic Weather Generator for
  Seasonal to Multi-Decadal Simulations".
  https://doi.org/10.1016/j.jhydrol.2015.12.036
  Core: daily GLMs conditional on areal seasonal rain/temperature covariates.
  The default normal-score/local model is a package variant; the explicit
  protocol follows the covariate structure with declared support changes.
* Eva M. Furrer and Richard W. Katz (2007), "Generalized Linear Modeling
  Approach to Stochastic Weather Generators". https://doi.org/10.3354/cr034129
  Background: daily occurrence/intensity regressions with seasonal covariates;
  links, lag predictors and extra weather variables here are explicit choices.
* William Kleiber, Richard W. Katz and Balaji Rajagopalan (2012), "Daily
  Spatiotemporal Precipitation Simulation Using Latent and Transformed Gaussian
  Processes". https://doi.org/10.1029/2011WR011105
  Core: latent occurrence and transformed amount Gaussian spatial fields;
  fitted isotropic kernels and finite Fourier fields are package variants.
* Andrew Verdin, Balaji Rajagopalan, William Kleiber, Guillermo Podesta and
  Federico Bert (2019), "BayGEN: A Bayesian Space-Time Stochastic Weather
  Generator". https://doi.org/10.1029/2017WR022473
  Inspiration: spatial GP coefficients. ``bayesian=True`` here is a Gaussian
  measurement/Laplace empirical-Bayes approximation to local GLM estimates,
  not BayGEN's posterior inference from daily data. The separate
  BayesianGLMWeatherGenerator implements a declared daily-likelihood extension.
* Jery R. Stedinger and Young-Oh Kim (2010), "Probabilities for Ensemble
  Forecasts Reflecting Climate Information".
  https://doi.org/10.1016/j.jhydrol.2010.06.038
  Background: forecast-distribution construction; the normal-score forecast
  density and optional transmission calibration here are package extensions.

Scientific references are method attributions, not software author metadata.
These models do not establish reproduction of the papers' published numerical
experiments or improved forecast skill without independent regional hindcasts.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize
from scipy.special import digamma, gammaincinv, gammaincc, gammaln, ndtr, ndtri, polygamma

from .conditioning import classify_seasons, normal_scores, tercile_normal_forecast
from .data import canonicalize_observations, prepare_probabilities, season_dates, seasonal_cube, validate_months
from .multivariate import VARIABLE_ORDER, _inverse_transform, _transform
from .spatial import GaussianSpatialField, IndependentField, _coordinates, fit_distance_model

STREAM_OCC, STREAM_AMT, STREAM_SEASON, STREAM_CONT = 200, 201, 299, 210


# ---------------------------------------------------------------------------
# Batched GLM estimation (sites x observations x covariates)
# ---------------------------------------------------------------------------
def _solve(A, b, ridge):
    p = A.shape[-1]
    return np.linalg.solve(A + ridge * np.eye(p), b[..., None])[..., 0]


def fit_probit(X, y, mask, ridge=1e-4, n_iter=40, tol=1e-8):
    """Probit GLM by IRLS for every site. X (S, N, p), y/mask (S, N).

    Returns coefficients (S, p) and their covariance (S, p, p) = inverse
    Fisher information (with a tiny ridge for sites without variation)."""
    S, N, p = X.shape
    m = mask.astype(float)
    frac = np.clip((y * m).sum(1) / np.maximum(m.sum(1), 1), 1e-3, 1 - 1e-3)
    beta = np.zeros((S, p))
    beta[:, 0] = ndtri(frac)
    for _ in range(n_iter):
        eta = np.clip(np.einsum("snp,sp->sn", X, beta), -6, 6)
        mu = np.clip(ndtr(eta), 1e-10, 1 - 1e-10)
        phi = np.maximum(np.exp(-0.5 * eta ** 2) / np.sqrt(2 * np.pi), 1e-12)
        w = m * phi ** 2 / (mu * (1 - mu))
        z = eta + (y - mu) / phi
        info = np.einsum("sn,snp,snq->spq", w, X, X)
        new = _solve(info, np.einsum("sn,snp,sn->sp", w, X, z), ridge)
        change = np.max(np.abs(new - beta))
        beta = new
        if change < tol:
            break
    eta = np.clip(np.einsum("snp,sp->sn", X, beta), -6, 6)
    mu = np.clip(ndtr(eta), 1e-10, 1 - 1e-10)
    phi = np.maximum(np.exp(-0.5 * eta ** 2) / np.sqrt(2 * np.pi), 1e-12)
    info = np.einsum("sn,snp,snq->spq", m * phi ** 2 / (mu * (1 - mu)), X, X)
    return beta, np.linalg.inv(info + ridge * np.eye(p))


def fit_gamma_log(X, y, mask, ridge=1e-6, n_iter=50, tol=1e-9):
    """Gamma GLM with log link (Fisher scoring) + shape MLE, per site.

    Returns beta (S, p), cov (S, p, p), shape k (S,), var(log k) (S,)."""
    S, N, p = X.shape
    m = mask.astype(float)
    yy = np.where(mask, y, 1.0)
    beta = np.zeros((S, p))
    beta[:, 0] = np.log(np.maximum((yy * m).sum(1) / np.maximum(m.sum(1), 1), 1e-3))
    XtX = np.einsum("sn,snp,snq->spq", m, X, X)
    for _ in range(n_iter):
        eta = np.clip(np.einsum("snp,sp->sn", X, beta), -10, 10)
        mu = np.exp(eta)
        z = eta + (yy - mu) / mu
        new = _solve(XtX, np.einsum("sn,snp,sn->sp", m, X, z), ridge)
        change = np.max(np.abs(new - beta))
        beta = new
        if change < tol:
            break
    mu = np.exp(np.clip(np.einsum("snp,sp->sn", X, beta), -10, 10))
    r = yy / mu
    n = np.maximum(m.sum(1), 1)
    s = np.maximum((m * (r - np.log(r) - 1)).sum(1) / n, 1e-6)
    k = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    for _ in range(30):
        k = np.maximum(k - (np.log(k) - digamma(k) - s) / (1 / k - polygamma(1, k)), 1e-3)
    cov = np.linalg.inv(XtX + ridge * np.eye(p)) / k[:, None, None]
    var_logk = 1.0 / np.maximum(n * (polygamma(1, k) - 1 / k) * k ** 2, 1e-12)
    return beta, cov, k, var_logk


def fit_linear(X, y, mask, wet, ridge=1e-6):
    """OLS per site with separate residual SD on wet and dry days."""
    S, N, p = X.shape
    m = mask.astype(float)
    XtX = np.einsum("sn,snp,snq->spq", m, X, X)
    beta = _solve(XtX, np.einsum("sn,snp,sn->sp", m, X, np.where(mask, y, 0.)), ridge)
    resid = np.where(mask, y - np.einsum("snp,sp->sn", X, beta), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # a state may have no days at a cell
        sd = np.stack([np.sqrt(np.nanmean(np.where(wet == st, resid, np.nan) ** 2, axis=1)) for st in (0, 1)], -1)
        pooled_sd = np.sqrt(np.nanmean(resid ** 2, axis=1))
    sd = np.where(np.isfinite(sd), sd, pooled_sd[:, None])
    s2 = pooled_sd ** 2
    cov = np.linalg.inv(XtX + ridge * np.eye(p)) * s2[:, None, None]
    n = np.maximum(m.sum(1), 2)
    return beta, cov, np.maximum(sd, 1e-6), resid, 1.0 / (2 * n)


def fit_gamma_truncated(X, y, mask, threshold, ridge=1e-6):
    """Raw Gamma law conditional on exceeding a strictly positive threshold.

    Package support-consistency extension to the Verdin et al. (2018) GLM
    framework, DOI 10.1016/j.jhydrol.2015.12.036, not a formula attributed to
    that paper. Full scientific authors and title are in the module References.

    Its log likelihood is log Gamma(y; k, mu/k) - log Q(k, k*threshold/mu).
    The normalization matters: fitting an untruncated Gamma on wet days and
    then suppressing sub-threshold simulations does not fit the simulated law.
    """
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    beta, cov, shape, variance = fit_gamma_log(X, y, mask, ridge=ridge)
    for s in range(X.shape[0]):
        good = mask[s] & np.isfinite(y[s]) & (y[s] > threshold)
        if good.sum() < max(3, X.shape[-1]):
            continue
        xs, ys = X[s, good], y[s, good]

        def nll(theta):
            k = np.exp(theta[-1])
            eta = xs @ theta[:-1]
            if np.any(np.abs(eta) > 50):
                return 1e100
            mu = np.exp(eta)
            survival = gammaincc(k, k * threshold / mu)
            if np.any(survival <= 0):
                return 1e100
            ll = (k * (np.log(k) - eta) - gammaln(k)
                  + (k - 1) * np.log(ys) - k * ys / mu - np.log(survival))
            return -ll.sum() + ridge * np.sum(theta[:-1] ** 2) / 2

        result = minimize(nll, np.r_[beta[s], np.log(shape[s])], method="L-BFGS-B",
                          bounds=[(None, None)] * X.shape[-1] + [(-7, 10)])
        if not result.success or not np.isfinite(result.fun):
            warnings.warn(f"Truncated Gamma fit did not converge at site {s}: {result.message}",
                          RuntimeWarning, stacklevel=2)
        if np.isfinite(result.fun):
            beta[s], shape[s] = result.x[:-1], np.exp(result.x[-1])
            covariance = np.asarray(result.hess_inv.todense())
            cov[s], variance[s] = covariance[:-1, :-1], covariance[-1, -1]
    return beta, cov, shape, variance


# ---------------------------------------------------------------------------
# Spatial Gaussian-process smoothing of coefficients (empirical-Bayes BayGEN)
# ---------------------------------------------------------------------------
def _chord(xyz_a, xyz_b):
    return np.sqrt(np.maximum(((xyz_a[:, None, :] - xyz_b[None, :, :]) ** 2).sum(-1), 0))


class CoefficientGP:
    """Independent GP prior m + GP(0, tau^2 exp(-d/rho)) per coefficient.

    Empirical-Bayes package approximation inspired by Verdin, Rajagopalan,
    Kleiber, Podesta and Bert (2019), BayGEN, DOI 10.1029/2017WR022473.
    It treats estimated coefficients as Gaussian measurements with estimated
    variances, rather than evaluating the daily-data posterior of that paper.

    ``fit(estimates, variances, lat, lon)``: the hyperparameters maximise the
    marginal likelihood of the per-cell estimates, whose sampling variances act
    as a known heteroscedastic nugget. ``posterior(lat, lon)`` returns the mean
    and covariance at any locations (training or new).
    """

    def __init__(self, n_hyper_sites=400, seed=0):
        self.n_hyper_sites, self.seed = int(n_hyper_sites), int(seed)

    def fit(self, estimates, variances, lat, lon):
        est = np.asarray(estimates, float)
        var = np.maximum(np.asarray(variances, float), 1e-10)
        good = np.isfinite(est).all(1) & np.isfinite(var).all(1)
        var = np.where(np.isfinite(var), var, 1e6)
        _, _, xyz = _coordinates(lat, lon)
        self.xyz, self.est, self.var, self.good = xyz, np.where(good[:, None], est, 0.), var, good
        rng = np.random.default_rng(self.seed)
        idx = np.flatnonzero(good)
        if idx.size == 0:
            raise ValueError("CoefficientGP needs at least one finite coefficient estimate with sampling variances")
        if idx.size > self.n_hyper_sites:
            idx = np.sort(rng.choice(idx, self.n_hyper_sites, replace=False))
        D = _chord(xyz[idx], xyz[idx])
        scale = np.median(D[D > 0]) if (D > 0).any() else 100.0
        self.hyper = []
        for j in range(est.shape[1]):
            b, v = est[idx, j], var[idx, j]
            spread = max(float(np.var(b)), 1e-8)

            def nll(theta):
                tau2, rho = np.exp(theta)
                C = tau2 * np.exp(-D / rho) + np.diag(v)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    return 1e12
                ones = np.linalg.solve(L, np.ones(len(b)))
                yb = np.linalg.solve(L, b)
                mean = (ones @ yb) / (ones @ ones)
                r = yb - mean * ones
                return 0.5 * r @ r + np.log(np.diag(L)).sum()

            if len(idx) < 3:
                self.hyper.append((float(np.mean(b)) if len(b) else 0., spread, scale))
                continue
            res = minimize(nll, np.log([spread, scale]), method="L-BFGS-B",
                           bounds=[(np.log(spread * 1e-4), np.log(spread * 10 + 1e-6)), (np.log(1.0), np.log(2e4))])
            tau2, rho = np.exp(res.x)
            C = tau2 * np.exp(-D / rho) + np.diag(v)
            Ci1 = np.linalg.solve(C, np.ones(len(b)))
            mean = float(Ci1 @ b / Ci1.sum())
            self.hyper.append((mean, float(tau2), float(rho)))
        return self

    def posterior(self, lat=None, lon=None, covariance=True):
        """Posterior mean (S*, P) and, optionally, covariance list [(S*, S*)] * P."""
        xyz_t = self.xyz if lat is None else _coordinates(lat, lon)[2]
        tr = np.flatnonzero(self.good)
        Dtt = _chord(self.xyz[tr], self.xyz[tr])
        Dst = _chord(xyz_t, self.xyz[tr])
        Dss = _chord(xyz_t, xyz_t) if covariance else None
        means, covs = [], []
        for j, (m, tau2, rho) in enumerate(self.hyper):
            C = tau2 * np.exp(-Dtt / rho) + np.diag(self.var[tr, j])
            Ks = tau2 * np.exp(-Dst / rho)
            L = np.linalg.cholesky(C + 1e-10 * np.eye(len(tr)))
            A = np.linalg.solve(L, Ks.T)                          # (n_tr, S*)
            alpha = np.linalg.solve(L, self.est[tr, j] - m)
            means.append(m + A.T @ alpha)
            if covariance:
                covs.append(tau2 * np.exp(-Dss / rho) - A.T @ A)
        return np.stack(means, -1), covs


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class GLMWeatherGenerator:
    """A declared GLM adaptation inspired by Verdin et al. (2018).

    Andrew Verdin, Balaji Rajagopalan, William Kleiber, Guillermo Podesta and
    Federico Bert (2018), DOI 10.1016/j.jhydrol.2015.12.036; the module
    References provide the full title and model lineage. The default local
    normal-score model differs from their areal physical-total covariates.
    ``paper_protocol='verdin_2018'`` follows the seasonal covariate structure,
    with a declared truncated-Gamma support extension, and does not reproduce
    the complete multi-season published experiment.
    ``bayesian=True`` adds an empirical-Bayes GP layer inspired by the same
    authors' BayGEN (2019), DOI 10.1029/2017WR022473; it is not full posterior
    sampling of the daily-data likelihood.

    Parameters
    ----------
    months, climatology, wet_threshold : as in :class:`WeatherGenerator`
    harmonics : number of annual harmonics in every linear predictor
    interaction : include z_S x wet_{t-1} in the occurrence model
    total_sampling : 'normal' draws z_S from N(mu, sigma) matched to the tercile
        forecast; 'tercile' draws a class from PB/PN/PA and a uniform position
        inside it (sampling "seasonal totals from the forecast" as Verdin et al.)
    spatial : 'distance' (fitted kernels, Gaussian fields) or 'independent'
    bayesian : GP smoothing of all coefficients (BayGEN-inspired empirical Bayes)
    parameter_uncertainty : in Bayesian mode, draw one coefficient field per member
    trace_rainfall : simulate sub-threshold rain on dry days (climatological)
    calibrate_covariate : deconvolve the covariate (see ``calibrate``). The
        historical z_S is a realised outcome rather than a perfect external
        predictor. An additional daily stochastic realization may change the
        transmission of its seasonal moments. Calibration is a package
        extension and requires independent checks of seasonal distributions;
        overdispersion and improved forecast skill are not universal results.
    seasonal_covariates : 'local_rainfall' preserves the original package
        model; 'domain_three' fits domain-mean seasonal rainfall for occurrence
        and amount, and domain-mean seasonal TMIN and TMAX for both temperature
        models. The latter requires both historical temperature variables.
    """

    def __init__(self, months=(7, 8, 9), climatology=(1991, 2020), wet_threshold=1.0, harmonics=2,
                 interaction=True, total_sampling="normal", spatial="distance", bayesian=False,
                 parameter_uncertainty=True, n_hyper_sites=400, n_features=256, max_pairs=300,
                 max_spatial_sites=64, trace_rainfall=True, seed=42, chunk_sites=256,
                 calibrate_covariate=True, calibration_members=300,
                 seasonal_covariates="local_rainfall", paper_protocol=None,
                 wet_rule="ge", amount_model="excess", rainfall_covariate="normal_score"):
        if paper_protocol not in {None, "verdin_2018"}:
            raise ValueError("paper_protocol must be None or 'verdin_2018'")
        if paper_protocol == "verdin_2018":
            seasonal_covariates, harmonics, interaction = "domain_three", 1, False
            wet_threshold, wet_rule, amount_model = .1, "gt", "raw_truncated"
            rainfall_covariate, trace_rainfall, calibrate_covariate = "standardized_total", False, False
            total_sampling = "tercile"
        if total_sampling not in {"normal", "tercile"}:
            raise ValueError("total_sampling must be 'normal' or 'tercile'")
        if spatial not in {"distance", "independent"}:
            raise ValueError("spatial must be 'distance' or 'independent'")
        if seasonal_covariates not in {"local_rainfall", "domain_three"}:
            raise ValueError("seasonal_covariates must be 'local_rainfall' or 'domain_three'")
        if wet_rule not in {"ge", "gt"} or amount_model not in {"excess", "raw_truncated"}:
            raise ValueError("wet_rule must be ge/gt and amount_model excess/raw_truncated")
        if rainfall_covariate not in {"normal_score", "standardized_total"}:
            raise ValueError("rainfall_covariate must be normal_score/standardized_total")
        if amount_model == "raw_truncated" and wet_rule != "gt":
            raise ValueError("raw_truncated amounts require wet_rule='gt'")
        self.paper_protocol, self.wet_rule = paper_protocol, wet_rule
        self.amount_model, self.rainfall_covariate = amount_model, rainfall_covariate
        self.months, self.climatology = validate_months(months), tuple(climatology)
        self.thr, self.harmonics, self.interaction = float(wet_threshold), int(harmonics), bool(interaction)
        self.total_sampling, self.spatial, self.bayesian = total_sampling, spatial, bool(bayesian)
        self.parameter_uncertainty, self.n_hyper_sites = bool(parameter_uncertainty), int(n_hyper_sites)
        self.n_features, self.max_pairs, self.max_spatial_sites = int(n_features), int(max_pairs), int(max_spatial_sites)
        self.trace_rainfall, self.seed, self.chunk_sites = bool(trace_rainfall), int(seed), int(chunk_sites)
        self.calibrate_covariate, self.calibration_members = bool(calibrate_covariate), int(calibration_members)
        self.seasonal_covariates = seasonal_covariates
        if not np.isfinite(self.thr) or self.thr <= 0:
            raise ValueError("wet_threshold must be finite and positive")
        if self.harmonics < 0 or self.chunk_sites < 1 or self.n_hyper_sites < 1:
            raise ValueError("harmonics must be nonnegative; chunk_sites and n_hyper_sites must be positive")
        if self.calibrate_covariate and self.calibration_members < 3:
            raise ValueError("calibration_members must be >= 3")

    # ---- design matrices -------------------------------------------------
    def _harm(self, dates):
        doy = pd.DatetimeIndex(dates).dayofyear.to_numpy().astype(float)
        cols = [np.ones_like(doy)]
        for k in range(1, self.harmonics + 1):
            cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
        return np.stack(cols, -1)                                    # (day, 1+2H)

    def _x_occ(self, H, lag, z):
        cols = [H, lag[..., None], z[..., None]]
        if self.interaction:
            cols.append((z * lag)[..., None])
        return np.concatenate(cols, -1)

    def _x_amt(self, H, lag, z):
        if self.paper_protocol == "verdin_2018":
            return np.concatenate([H, z[..., None]], -1)
        return np.concatenate([H, lag[..., None], z[..., None]], -1)

    def _x_cont(self, H, wet, lagx, z):
        return np.concatenate([H, wet[..., None], lagx[..., None], z[..., None]], -1)

    def _x_temperature(self, H, wet, lag_min, lag_max, z_min, z_max, trend):
        """Optional Verdin-style daily temperature predictor, in Celsius."""
        return np.concatenate([H, wet[..., None], lag_min[..., None], lag_max[..., None],
                               z_min[..., None], z_max[..., None], trend[..., None]], -1)

    # ---- fit -------------------------------------------------------------
    def fit(self, observations: xr.Dataset):
        """Fit per-cell GLMs on every complete season (no forecast needed)."""
        obs = canonicalize_observations(observations)
        cube = seasonal_cube(obs, months=self.months).load()
        self.y_, self.x_ = obs.Y.values, obs.X.values
        lat, lon = np.meshgrid(self.y_, self.x_, indexing="ij")
        self.lat_, self.lon_ = lat.ravel(), lon.ravel()
        self.years_ = cube.season_year.values.astype(int)
        self.month_ = cube.month.values.astype(int)
        ny, nd = cube.sizes["season_year"], cube.sizes["day"]
        ns = self.lat_.size
        vals = {v: cube[v].transpose("season_year", "day", "Y", "X").values.reshape(ny, nd, ns) for v in cube.data_vars}
        rain = vals.pop("PRCP")
        self.variables_ = tuple(v for v in VARIABLE_ORDER if v in vals)
        if self.seasonal_covariates == "domain_three" and not {"TMIN", "TMAX"}.issubset(vals):
            raise ValueError("seasonal_covariates='domain_three' requires observed TMIN and TMAX")
        if self.seasonal_covariates == "domain_three":
            # A changing set of reporting sites would produce a false
            # interannual domain anomaly. Require complete seasons on one
            # fixed support; an entirely masked cell stays outside the domain.
            covariate_fields = (rain, vals["TMIN"], vals["TMAX"])
            active = np.any([np.isfinite(field).any(axis=(0, 1))
                             for field in covariate_fields], axis=0)
            complete = np.all([np.isfinite(field).all(axis=(0, 1))
                               for field in covariate_fields], axis=0)
            if not complete.any() or np.any(active & ~complete):
                raise ValueError("domain_three requires complete PRCP, TMIN and TMAX seasons "
                                 "on a fixed spatial support; remove or fill incomplete cells first")
            self.domain_support_ = complete
        self.units_ = {v: obs[v].attrs.get("units", "") for v in obs.data_vars}
        # day before each season: lag covariates
        stacked = obs.stack(site=("Y", "X")).transpose("T", "site")
        pre_dates = [season_dates(int(y), self.months)[0] - pd.Timedelta(days=1) for y in self.years_]
        pre = stacked.reindex(T=pd.DatetimeIndex(pre_dates)).load()
        pre_rain = pre["PRCP"].values                                 # (year, site)
        # The legacy local mode uses a different seasonal rain covariate at
        # each cell. The domain mode uses one areal season covariate across the
        # fitted domain, as in Verdin et al. (2018), retaining the package's
        # normal-score transformation for precipitation.
        totals = np.where(np.isfinite(rain).all(1), np.nansum(rain, 1), np.nan)
        if self.seasonal_covariates == "domain_three":
            area_weight = np.maximum(np.cos(np.deg2rad(self.lat_)), 0.0)[self.domain_support_]
            if area_weight.sum() <= 0:
                raise ValueError("domain_three requires positive cosine-latitude area weights")
            area_totals = np.average(totals[:, self.domain_support_],
                                     weights=area_weight, axis=1)
            ref_mask = (self.years_ >= self.climatology[0]) & (self.years_ <= self.climatology[1])
            if ref_mask.sum() < 3:
                raise ValueError("domain_three requires at least three complete climatology seasons")
            self.ref_domain_totals_ = np.sort(area_totals[ref_mask])
            center, scale = np.mean(area_totals[ref_mask]), np.std(area_totals[ref_mask], ddof=1)
            if self.rainfall_covariate == "standardized_total" and scale <= 0:
                raise ValueError("standardized_total requires varying climatological seasonal totals")
            self.rainfall_reference_ = (float(center), float(max(scale, 1e-12)))
            score = ((area_totals - center) / scale if self.rainfall_covariate == "standardized_total"
                     else normal_scores(area_totals[:, None], self.years_, climatology=self.climatology)[:, 0])
            self.z_hist_ = np.broadcast_to(score[:, None], (ny, ns)).copy()
        else:
            if self.rainfall_covariate != "normal_score":
                raise ValueError("standardized_total currently requires seasonal_covariates='domain_three'")
            self.z_hist_ = normal_scores(totals, self.years_, climatology=self.climatology)
        _, self.thresholds_ = classify_seasons(totals, self.years_, climatology=self.climatology)
        ref = (self.years_ >= self.climatology[0]) & (self.years_ <= self.climatology[1])
        self.ref_totals_ = np.sort(np.where(np.isfinite(totals[ref]), totals[ref], np.nan), axis=0)
        self.temperature_reference_, self.temperature_z_hist_ = {}, {}
        self.year_center_ = float(np.mean(self.years_))
        self.year_scale_ = max(float(np.max(np.abs(self.years_ - self.year_center_))), 1.0)
        if self.seasonal_covariates == "domain_three":
            for v in ("TMIN", "TMAX"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    # Average daily values into a seasonal mean at each site,
                    # then use the same cos(latitude) domain weights as the
                    # gridded forecast covariate at simulation time.
                    site_means = np.mean(vals[v][:, :, self.domain_support_], axis=1)
                    annual = np.average(site_means, weights=area_weight, axis=1)
                    center = float(np.nanmean(annual[ref]))
                    scale = float(np.nanstd(annual[ref], ddof=1))
                if np.sum(np.isfinite(annual[ref])) < 3 or not np.isfinite(scale) or scale <= 0:
                    raise ValueError(f"domain_three requires at least three varying reference seasons for {v}")
                self.temperature_reference_[v] = (center, scale)
                self.temperature_z_hist_[v] = (annual - center) / scale
        H = self._harm(season_dates(int(self.years_[0]), self.months))  # (nd, h)
        self.H_ = H
        compare = np.greater if self.wet_rule == "gt" else np.greater_equal
        wet = np.where(np.isfinite(rain), compare(rain, self.thr).astype(float), np.nan)
        lag = np.concatenate([np.where(np.isfinite(pre_rain), compare(pre_rain, self.thr).astype(float), np.nan)[:, None], wet[:, :-1]], 1)
        z = np.broadcast_to(self.z_hist_[:, None, :], rain.shape)
        # (site, N) layout
        flat = lambda a: np.moveaxis(a, -1, 0).reshape(ns, ny * nd)
        # Gregorian day-of-year differs after February in leap years. The
        # historical design must use each season's own dates, as generation does.
        Hn = np.stack([self._harm(season_dates(int(y), self.months))
                       for y in self.years_]).reshape(ny * nd, -1)
        Y_wet, L_wet, Z = flat(wet), flat(lag), flat(z)
        ok_occ = np.isfinite(Y_wet) & np.isfinite(L_wet) & np.isfinite(Z)
        excess = flat(rain) - (self.thr if self.amount_model == "excess" else 0.)
        ok_amt = ok_occ & (Y_wet == 1)
        p_occ = H.shape[1] + 2 + int(self.interaction)
        p_amt = H.shape[1] + (1 if self.paper_protocol == "verdin_2018" else 2)
        self.b_occ_, self.c_occ_ = np.zeros((ns, p_occ)), np.zeros((ns, p_occ, p_occ))
        self.b_amt_, self.c_amt_ = np.zeros((ns, p_amt)), np.zeros((ns, p_amt, p_amt))
        self.shape_, self.var_logk_ = np.ones(ns), np.ones(ns)
        cont = {}
        for s0 in range(0, ns, self.chunk_sites):
            sl = slice(s0, min(ns, s0 + self.chunk_sites))
            Hs = np.broadcast_to(Hn[None], (sl.stop - sl.start,) + Hn.shape)
            Xo = self._x_occ(Hs, np.nan_to_num(L_wet[sl]), np.nan_to_num(Z[sl]))
            self.b_occ_[sl], self.c_occ_[sl] = fit_probit(Xo, np.nan_to_num(Y_wet[sl]), ok_occ[sl])
            Xa = self._x_amt(Hs, np.nan_to_num(L_wet[sl]), np.nan_to_num(Z[sl]))
            if self.amount_model == "raw_truncated":
                b, c, k, vk = fit_gamma_truncated(Xa, np.nan_to_num(excess[sl]), ok_amt[sl], self.thr)
            else:
                b, c, k, vk = fit_gamma_log(Xa, np.maximum(np.nan_to_num(excess[sl]), 0.01), ok_amt[sl])
            self.b_amt_[sl], self.c_amt_[sl], self.shape_[sl], self.var_logk_[sl] = b, c, k, vk
        # other variables: transformed linear models with AR(1) and wet/dry effect
        self.b_cont_, self.c_cont_, self.sd_cont_, self.var_logsd_, resid = {}, {}, {}, {}, {}
        self.init_cont_ = {}
        if self.seasonal_covariates == "domain_three":
            temp_lags = {}
            for name in ("TMIN", "TMAX"):
                temp_hist = _transform(name, vals[name])
                temp_pre = _transform(name, pre[name].values)
                temp_lags[name] = flat(np.concatenate([temp_pre[:, None], temp_hist[:, :-1]], 1))
            temp_z = {name: flat(np.broadcast_to(self.temperature_z_hist_[name][:, None, None], rain.shape))
                      for name in ("TMIN", "TMAX")}
            year_trend = flat(np.broadcast_to(((self.years_ - self.year_center_) / self.year_scale_)[:, None, None],
                                                   rain.shape))
        for v in self.variables_:
            g = _transform(v, vals[v])
            gpre = _transform(v, pre[v].values)
            glag = np.concatenate([gpre[:, None], g[:, :-1]], 1)
            Gy, Gl = flat(g), flat(glag)
            temp_mode = self.seasonal_covariates == "domain_three" and v in ("TMIN", "TMAX")
            ok = ok_occ & np.isfinite(Gy) & np.isfinite(Gl)
            if temp_mode:
                ok &= (np.isfinite(temp_lags["TMIN"]) & np.isfinite(temp_lags["TMAX"])
                       & np.isfinite(temp_z["TMIN"]) & np.isfinite(temp_z["TMAX"]))
            p_c = H.shape[1] + (6 if temp_mode else 3)
            B, C = np.zeros((ns, p_c)), np.zeros((ns, p_c, p_c))
            SD, R = np.ones((ns, 2)), np.full((ns, ny * nd), np.nan)
            VL = np.ones(ns)
            for s0 in range(0, ns, self.chunk_sites):
                sl = slice(s0, min(ns, s0 + self.chunk_sites))
                Hs = np.broadcast_to(Hn[None], (sl.stop - sl.start,) + Hn.shape)
                if temp_mode:
                    Xc = self._x_temperature(
                        Hs, np.nan_to_num(Y_wet[sl]),
                        np.nan_to_num(temp_lags["TMIN"][sl]), np.nan_to_num(temp_lags["TMAX"][sl]),
                        np.nan_to_num(temp_z["TMIN"][sl]), np.nan_to_num(temp_z["TMAX"][sl]),
                        year_trend[sl])
                else:
                    Xc = self._x_cont(Hs, np.nan_to_num(Y_wet[sl]), np.nan_to_num(Gl[sl]), np.nan_to_num(Z[sl]))
                B[sl], C[sl], SD[sl], R[sl], VL[sl] = fit_linear(Xc, np.nan_to_num(Gy[sl]), ok[sl], np.nan_to_num(Y_wet[sl]))
            self.b_cont_[v], self.c_cont_[v], self.sd_cont_[v], self.var_logsd_[v] = B, C, SD, VL
            resid[v] = R / np.where(np.nan_to_num(Y_wet) == 1, SD[:, 1:2], SD[:, 0:1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                init = np.nanmean(gpre, axis=0)
                # no pre-season record (e.g. files holding only the season): use day 1
                self.init_cont_[v] = np.where(np.isfinite(init), init, np.nanmean(g[:, 0], axis=0))
        # cross-variable residual correlation per site (Cholesky)
        nv = len(self.variables_)
        self.chol_cont_ = np.tile(np.eye(max(nv, 1)), (ns, 1, 1))
        self.chol_cont_month_ = None
        if nv > 1:
            E = np.stack([resid[v] for v in self.variables_], -1)       # (site, N, nv)
            okE = np.isfinite(E).all(-1)
            for s in range(ns):
                e = E[s][okE[s]]
                if len(e) > 3 * nv:
                    # Constant variables have no estimable residual correlation.
                    # Keep identity rather than passing NaNs to eigh.
                    if np.any(np.std(e, axis=0) <= 1e-12):
                        continue
                    corr = np.corrcoef(e.T)
                    if not np.isfinite(corr).all():
                        continue
                    w_, V_ = np.linalg.eigh((corr + corr.T) / 2)
                    corr = (V_ * np.maximum(w_, 1e-6)) @ V_.T
                    d = np.sqrt(np.diag(corr))
                    self.chol_cont_[s] = np.linalg.cholesky(corr / d[:, None] / d[None, :])
            if self.seasonal_covariates == "domain_three":
                self.chol_cont_month_ = np.tile(self.chol_cont_[None], (12, 1, 1, 1))
                month_flat = np.tile(self.month_, ny)
                for month_number in np.unique(self.month_):
                    mask = month_flat == month_number
                    for s in range(ns):
                        e = E[s, mask][okE[s, mask]]
                        if len(e) <= 3 * nv or np.any(np.std(e, axis=0) <= 1e-12):
                            continue
                        corr = np.corrcoef(e.T)
                        if not np.isfinite(corr).all():
                            continue
                        w_, V_ = np.linalg.eigh((corr + corr.T) / 2)
                        corr = (V_ * np.maximum(w_, 1e-6)) @ V_.T
                        d = np.sqrt(np.diag(corr))
                        self.chol_cont_month_[month_number - 1, s] = np.linalg.cholesky(corr / d[:, None] / d[None, :])
        # trace rainfall (climatological, per month) and first-day wet probability
        mon = self.month_
        self.trace_p_ = np.zeros((12, ns)); self.trace_m_ = np.zeros((12, ns))
        for mth in np.unique(mon):
            x = rain[:, mon == mth]
            dry = np.isfinite(x) & ~compare(x, self.thr)
            tr = dry & (x > 0)
            self.trace_p_[mth - 1] = tr.sum((0, 1)) / np.maximum(dry.sum((0, 1)), 1)
            self.trace_m_[mth - 1] = np.where(tr, x, 0).sum((0, 1)) / np.maximum(tr.sum((0, 1)), 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pw = np.nanmean(np.where(np.isfinite(pre_rain), compare(pre_rain, self.thr), np.nan), axis=0)
            self.p_first_wet_ = np.where(np.isfinite(pw), pw, np.nanmean(wet[:, 0], axis=0))
        self.valid_ = (np.isfinite(self.z_hist_).any(0) & np.isfinite(self.b_occ_).all(1)
                       & np.isfinite(totals).any(0))
        # Bayesian GP layer
        self._pack_layout()
        if self.bayesian:
            est, var = self._pack()
            self.gp_ = CoefficientGP(self.n_hyper_sites, self.seed).fit(est, var, self.lat_, self.lon_)
        # spatial kernels
        self.spatial_models_ = {}
        if self.spatial == "distance" and ns > 1:
            sample = np.unique(np.linspace(0, ns - 1, min(ns, self.max_spatial_sites)).astype(int))
            fits = {"glm_occurrence": (Y_wet.T, "power", "binary", 30),
                    "glm_amount": (np.where(ok_amt, excess, np.nan).T, "exponential", "gaussian", 30),
                    "glm_season": (self.z_hist_, "exponential", "none", 10)}
            for v in self.variables_:
                fits[f"glm_{v}"] = (resid[v].T, "exponential", "none", 30)
            for key, (data, kind, transform, overlap) in fits.items():
                self.spatial_models_[key] = fit_distance_model(
                    np.asarray(data)[:, sample], self.lat_[sample], self.lon_[sample], kind=kind,
                    max_pairs=self.max_pairs, seed=self.seed, transform=transform, min_overlap=overlap)
            if self.seasonal_covariates == "domain_three":
                # Daily spatial residual covariance is estimated separately
                # by calendar month; sparse pairs fall back within the
                # distance estimator rather than borrowing a future month.
                month_flat = np.tile(self.month_, ny)
                for month_number in np.unique(self.month_):
                    selector = month_flat == month_number
                    for key, (data, kind, transform, overlap) in fits.items():
                        if key == "glm_season":
                            continue
                        monthly = np.asarray(data)[selector][:, sample]
                        self.spatial_models_[f"{key}_month_{month_number}"] = fit_distance_model(
                            monthly, self.lat_[sample], self.lon_[sample], kind=kind,
                            max_pairs=self.max_pairs, seed=self.seed, transform=transform, min_overlap=overlap)
        self.transmission_ = None
        if self.calibrate_covariate:
            self.calibrate(self.calibration_members)
        self.diagnostics_ = {
            "model": "Verdin-type GLM: probit occurrence, Gamma(log) amounts, Gaussian lagged other variables",
            "covariate": ("z_S = normal score of domain-mean season total; TMIN/TMAX domain-mean standardized "
                          "seasonal temperature covariates and cross-variable temperature lags"
                          if self.seasonal_covariates == "domain_three" else
                          "z_S = normal score of the local season total (reference-period climatology)"),
            "bayesian": ("empirical-Bayes GP on coefficients (BayGEN-type), hyperparameters "
                         + str([tuple(round(x, 4) for x in h) for h in self.gp_.hyper]) if self.bayesian else "off"),
            "occurrence_z_coefficient_mean": float(np.nanmean(self.b_occ_[:, self.H_.shape[1] + 1])),
            "amount_z_coefficient_mean": float(np.nanmean(self.b_amt_[:, -1])),
            "training_years": self.years_.tolist(),
            "paper_protocol": self.paper_protocol,
            "wet_rule": self.wet_rule,
            "amount_model": self.amount_model,
            "rainfall_covariate": self.rainfall_covariate,
        }
        if self.rainfall_covariate == "standardized_total":
            self.diagnostics_["covariate"] = (
                "Affine standardized physical domain seasonal rain total; TMIN/TMAX "
                "domain seasonal means, cross-temperature lags and year trend")
        if self.seasonal_covariates == "domain_three":
            self.diagnostics_["domain_support_sites"] = int(self.domain_support_.sum())
            self.diagnostics_["rain_probability_aggregation"] = (
                "Area-weighted mean of local tercile probabilities is an approximation, "
                "not the probability distribution of the domain-mean seasonal total")
        return self

    # ---- parameter packing for the GP layer ---------------------------------
    def _pack_layout(self):
        self._layout = [("occ", self.b_occ_.shape[1]), ("amt", self.b_amt_.shape[1]), ("logk", 1)]
        for v in self.variables_:
            self._layout += [(f"c_{v}", self.b_cont_[v].shape[1]), (f"logsd_{v}", 2)]

    def _pack(self):
        est = [self.b_occ_, self.b_amt_, np.log(self.shape_)[:, None]]
        var = [np.diagonal(self.c_occ_, axis1=1, axis2=2), np.diagonal(self.c_amt_, axis1=1, axis2=2), self.var_logk_[:, None]]
        for v in self.variables_:
            est += [self.b_cont_[v], np.log(self.sd_cont_[v])]
            var += [np.diagonal(self.c_cont_[v], axis1=1, axis2=2), np.repeat(self.var_logsd_[v][:, None], 2, 1)]
        estimates, variances = np.concatenate(est, 1), np.concatenate(var, 1)
        # Empty/insufficient reference cells are not coefficient observations
        # for the Bayesian spatial fit.
        estimates[~self.valid_] = np.nan
        variances[~self.valid_] = np.nan
        return estimates, variances

    def _unpack(self, theta):
        """theta (..., S, P) -> dict of parameter arrays with the same leading dims."""
        out, i = {}, 0
        for name, n in self._layout:
            out[name] = theta[..., i:i + n]
            i += n
        return out

    def _parameters(self, n_members, member_start, target_latlon):
        """Parameter dict with leading member axis (n or 1) at the simulation cells."""
        if not self.bayesian:
            if target_latlon is not None:
                raise ValueError("target grids need bayesian=True (GP prediction of coefficients)")
            est, _ = self._pack()
            return self._unpack(est[None])
        lat, lon = target_latlon if target_latlon is not None else (None, None)
        mean, covs = self.gp_.posterior(lat, lon, covariance=self.parameter_uncertainty)
        if not self.parameter_uncertainty:
            return self._unpack(mean[None])
        ns, P = mean.shape
        Ls = [np.linalg.cholesky(c + 1e-8 * np.eye(ns)) for c in covs]
        theta = np.empty((n_members, ns, P))
        for i, m in enumerate(range(member_start, member_start + n_members)):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 5151, int(m)]))
            e = rng.standard_normal((P, ns))
            theta[i] = mean + np.stack([Ls[j] @ e[j] for j in range(P)], -1)
        return self._unpack(theta)

    # ---- covariate transmission calibration ------------------------------------
    def calibrate(self, n_members=300):
        """Estimate how the covariate reaches the generated total, per cell.

        This two-run transmission/deconvolution model is a package extension,
        not a calibration algorithm attributed to Verdin et al. (2018/2019).

        Two simulations with a climatological covariate, z ~ N(0, 1) and
        z ~ N(0, 1/4), give the standardized generated total (reference mean and
        SD). From them:
        the mean response slope a and offset b (regression in the first run),
        and the variance response Var_out(s^2) = nu + A s^2 (two-point fit;
        nu is the weather-noise variance the daily simulation adds by itself).
        ``generate`` then feeds the deconvolved covariate with mean (m - b)/a and
        variance max(s^2 - nu, 0.05 s^2)/A, where m and s^2 are the target
        (forecast) mean and variance. This targets those moments under an
        approximate affine mean/variance response fitted to finite simulations.
        Nonlinear links, clipping, finite Monte Carlo error and the positive
        variance floor prevent a general equality guarantee. A requested
        variance below nu cannot be attained by this noise model. Validate
        generated seasonal distributions independently after calibration.
        """
        n_members = int(n_members)
        if n_members < 3:
            raise ValueError("n_members must be >= 3 for transmission calibration")
        self.transmission_ = None
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 777]))
        ns = self.lat_.size
        runs = []
        for k, scale in enumerate((1.0, 0.5)):
            z = scale * rng.standard_normal((n_members, ns))
            if self.seasonal_covariates == "domain_three":
                z = np.broadcast_to(z[:, :1], z.shape).copy()
            sim = self.generate(int(self.years_.max()) + 1, n_members, covariate=z, calibrated_covariate=False,
                                _seed=self.seed + 7919 + k)
            runs.append((z, sim.PRCP.sum("T", skipna=False).values.reshape(n_members, ns)))
        if self.seasonal_covariates == "domain_three":
            ref = self.ref_domain_totals_[np.isfinite(self.ref_domain_totals_)]
            weights = np.maximum(np.cos(np.deg2rad(self.lat_)), 0.)
            aggregate = []
            for z, total in runs:
                good = np.isfinite(total)
                denominator = np.sum(good * weights[None], axis=1)
                domain_total = np.divide(np.nansum(total * weights[None], axis=1), denominator,
                                         out=np.full(n_members, np.nan), where=denominator > 0)
                aggregate.append((z[:, 0], domain_total))
            if (ref.size >= 3 and ref.std(ddof=1) > 1e-12 * max(abs(ref.mean()), 1.)
                    and all(np.isfinite(t).all() for _, t in aggregate)):
                z1, zg1 = aggregate[0][0], (aggregate[0][1] - ref.mean()) / ref.std(ddof=1)
                zg2 = (aggregate[1][1] - ref.mean()) / ref.std(ddof=1)
                slope, icpt = np.polyfit(z1, zg1, 1)
                v1, v2 = zg1.var(), zg2.var()
                amp = max((v1 - v2) / .75, .05)
                noise = max(v1 - amp, 0.)
                scalar = (max(float(slope), .05), float(icpt), float(noise), float(amp))
            else:
                scalar = (1., 0., 0., 1.)
            self.transmission_ = {key: np.full(ns, value)
                                  for key, value in zip(("a", "b", "nu", "A"), scalar)}
            return self.transmission_
        a, b, nu, A = np.ones(ns), np.zeros(ns), np.zeros(ns), np.ones(ns)
        for s in range(ns):
            ref = self.ref_totals_[:, s][np.isfinite(self.ref_totals_[:, s])]
            if (ref.size < 3 or ref.std(ddof=1) <= 1e-12 * max(abs(ref.mean()), 1.)
                    or not all(np.isfinite(t[:, s]).all() for _, t in runs)):
                continue
            # standardized totals (not saturating rank scores): the variance is
            # then matched in mm, including tails beyond the reference range
            z1, zg1 = runs[0][0][:, s], (runs[0][1][:, s] - ref.mean()) / ref.std(ddof=1)
            zg2 = (runs[1][1][:, s] - ref.mean()) / ref.std(ddof=1)
            slope, icpt = np.polyfit(z1, zg1, 1)
            v1, v2 = zg1.var(), zg2.var()
            A[s] = max((v1 - v2) / 0.75, 0.05)
            nu[s] = max(v1 - A[s], 0.0)
            a[s], b[s] = max(slope, 0.05), icpt
        self.transmission_ = {"a": a, "b": b, "nu": nu, "A": A}
        return self.transmission_

    def totals_to_covariate(self, totals):
        """Fitted precipitation covariates of seasonal totals on the training grid.

        The default is a normal score. ``rainfall_covariate='standardized_total'``
        instead uses (physical domain total - reference mean) / reference SD,
        an affine transformation that allows physical-total extrapolation.

        Use it to drive the GLM with a dynamical model: bias-correct its daily
        ensemble (``was_disaggregation.dynamical``), sum each member's season, convert
        the totals here, and pass them as ``generate(covariate=...)``. No tercile
        step is involved. In ``domain_three`` mode, a vector of already
        domain-averaged totals (mm/member) is also accepted; gridded totals
        are first area averaged over the same fixed historical support.
        """
        if isinstance(totals, xr.DataArray) and set(totals.dims) == {"member", "Y", "X"}:
            if not (np.array_equal(totals.Y.values, self.y_) and np.array_equal(totals.X.values, self.x_)):
                raise ValueError("totals Y,X coordinates must match the training grid")
            totals = totals.transpose("member", "Y", "X")
        t = np.asarray(totals.values if isinstance(totals, xr.DataArray) else totals, float)
        if self.seasonal_covariates == "domain_three":
            if t.ndim == 1:
                domain_total = t
            else:
                if t.ndim < 2 or t.reshape(t.shape[0], -1).shape[1] != self.lat_.size:
                    raise ValueError("domain_three totals must be (member,) or match the training grid")
                fields = t.reshape(t.shape[0], -1)
                if not np.isfinite(fields[:, self.domain_support_]).all():
                    raise ValueError("domain_three totals must cover the fixed domain support")
                weight = np.maximum(np.cos(np.deg2rad(self.lat_[self.domain_support_])), 0.)
                domain_total = np.average(fields[:, self.domain_support_], weights=weight, axis=1)
            ref = self.ref_domain_totals_[np.isfinite(self.ref_domain_totals_)]
            if self.rainfall_covariate == "standardized_total":
                center, scale = self.rainfall_reference_
                score = (domain_total - center) / scale
            else:
                score = self._score(domain_total, ref)
            return np.broadcast_to(score[:, None], (len(score), self.lat_.size)).copy()
        if t.ndim < 2:
            raise ValueError("totals must have a member axis and the training-site dimensions")
        t = t.reshape(t.shape[0], -1)
        if t.shape[1] != self.lat_.size:
            raise ValueError("totals must match the number of training sites")
        out = np.full(t.shape, np.nan)
        for s in range(t.shape[1]):
            ref = self.ref_totals_[:, s][np.isfinite(self.ref_totals_[:, s])]
            if ref.size >= 3:
                out[:, s] = self._score(t[:, s], ref)
        return out

    @staticmethod
    def _score(values, ref):
        below = np.searchsorted(ref, values, side="left")
        equal = np.searchsorted(ref, values, side="right") - below
        scores = ndtri(np.clip((below + 0.5 * equal + 0.5) / (ref.size + 1), 1e-4, 1 - 1e-4))
        return np.where(np.isfinite(values), scores, np.nan)

    def _deconvolve(self, z, m, s2, nearest):
        if self.transmission_ is None:
            return z
        a, b, nu, A = (self.transmission_[k][nearest][None] for k in ("a", "b", "nu", "A"))
        m = np.asarray(m, float)[None] if np.ndim(m) == 1 else np.asarray(m, float)
        s2 = np.asarray(s2, float)[None] if np.ndim(s2) == 1 else np.asarray(s2, float)
        var_in = np.maximum(s2 - nu, 0.05 * s2) / A
        return (m - b) / a + (z - m) * np.sqrt(var_in / np.maximum(s2, 1e-9))

    # ---- seasonal covariate from the forecast ---------------------------------
    def _target_moments(self, p):
        """Mean and variance of z_S under the chosen total_sampling scheme."""
        if self.total_sampling == "normal" and self.rainfall_covariate == "normal_score":
            mu, sigma = tercile_normal_forecast(p)
            return mu, sigma ** 2
        u = (np.arange(4000) + 0.5) / 4000
        zz = self.sample_covariate(p, ndtri(u)[:, None] * np.ones((1, p.shape[1])), raw=True)
        return zz.mean(0), zz.var(0)

    def sample_covariate(self, probabilities, eps, raw=False):
        """Fitted rain covariates from tercile probabilities and N(0,1) fields."""
        p = np.asarray(probabilities, float)
        if self.total_sampling == "normal":
            mu, sigma = tercile_normal_forecast(p)
            score = mu[None] + sigma[None] * eps
        else:
            u = np.clip(ndtr(eps), 1e-9, 1 - 1e-9)
            c1, c2 = p[0][None], (p[0] + p[1])[None]
            cls = (u > c1).astype(int) + (u > c2).astype(int)
            lo = np.where(cls == 0, 0, np.where(cls == 1, c1, c2))
            width = np.where(cls == 0, p[0][None], np.where(cls == 1, p[1][None], p[2][None]))
            pos = np.clip((u - lo) / np.maximum(width, 1e-9), 1e-6, 1 - 1e-6)
            score = ndtri((cls + pos) / 3.0)
        return self._normal_to_rain_covariate(score)

    def _normal_to_rain_covariate(self, score):
        if self.rainfall_covariate == "normal_score":
            return score
        ref = self.ref_domain_totals_
        # A continuous empirical inverse is a documented tail limitation: it
        # clips at observed extrema. Direct seasonal_total permits extrapolation.
        q = (np.arange(ref.size) + .5) / ref.size
        total = np.interp(ndtr(score), q, ref)
        center, scale = self.rainfall_reference_
        return (total - center) / scale

    def _temperature_scores(self, seasonal_temperature, n_members, yy, xx):
        """Convert user seasonal Celsius forecasts to domain-wide anomaly scores.

        Spatial fields, when provided, are reduced to an area-weighted domain
        average before entering the temperature predictors. Supplying an
        ensemble vector keeps the user's rain/temperature member pairing.
        """
        if self.seasonal_covariates != "domain_three":
            if seasonal_temperature is not None:
                raise ValueError("seasonal_temperature requires seasonal_covariates='domain_three'")
            return {}, {}
        mapping = {} if seasonal_temperature is None else seasonal_temperature
        if not isinstance(mapping, dict) or set(mapping) - {"TMIN", "TMAX"}:
            raise ValueError("seasonal_temperature must map TMIN/TMAX to seasonal means in degrees Celsius")
        ns = len(yy) * len(xx)
        grid_weights = np.maximum(np.cos(np.deg2rad(np.repeat(yy, len(xx)))), 0.)
        support = (self.domain_support_ if np.array_equal(yy, self.y_)
                   and np.array_equal(xx, self.x_) else np.ones(ns, dtype=bool))
        grid_weights = grid_weights[support]
        if grid_weights.sum() <= 0:
            raise ValueError("seasonal_temperature requires positive cosine-latitude area weights")
        score, physical = {}, {}
        for name in ("TMIN", "TMAX"):
            center, scale = self.temperature_reference_[name]
            raw = mapping.get(name, center)
            if isinstance(raw, xr.DataArray):
                if {"Y", "X"}.issubset(raw.dims):
                    if (not np.array_equal(raw.Y.values, yy) or not np.array_equal(raw.X.values, xx)
                            or set(raw.dims) - {"member", "Y", "X"}):
                        raise ValueError(f"{name} seasonal temperature grid must match the simulation Y,X")
                    raw = raw.transpose(*(("member", "Y", "X") if "member" in raw.dims else ("Y", "X"))).values
                else:
                    if set(raw.dims) - {"member"}:
                        raise ValueError(f"{name} seasonal temperature may have only member,Y,X dimensions")
                    raw = raw.values
            x = np.asarray(raw, dtype=float)
            if x.ndim == 0:
                x = np.full(n_members, x.item())
            elif x.shape == (n_members,):
                pass
            elif x.shape == (len(yy), len(xx)):
                x = x.reshape(1, ns)
            elif x.shape == (n_members, len(yy), len(xx)):
                x = x.reshape(n_members, ns)
            elif x.shape != (n_members, ns):
                raise ValueError(f"{name} must be a scalar, (member,), (Y,X), or (member,Y,X)")
            if x.ndim == 2:
                if not np.isfinite(x[:, support]).all():
                    raise ValueError(f"{name} seasonal temperature forecast must cover the fixed domain support")
                x = np.average(x[:, support], weights=grid_weights, axis=1)
                if x.size == 1:
                    x = np.repeat(x, n_members)
            if x.shape != (n_members,) or not np.isfinite(x).all():
                raise ValueError(f"{name} seasonal temperature forecast needs finite values for every member")
            physical[name] = x
            score[name] = np.broadcast_to(((x - center) / scale)[:, None], (n_members, ns))
        return score, physical

    # ---- simulation ---------------------------------------------------------
    def generate(self, year, n_members=20, probabilities=None, covariate=None, member_start=0, target=None,
                 calibrated_covariate=True, _seed=None, seasonal_temperature=None,
                 seasonal_total=None, domain_probabilities=None):
        """Simulate one season.

        probabilities : DataArray (probability, Y, X), local tercile forecasts.
            In 'domain_three' mode their area-weighted mean is used as an
            approximate forecast for the domain total. The mean of local
            tercile probabilities is generally not the probability distribution
            of the domain-mean seasonal total. For a directly forecast domain
            total, convert its members with the historical domain-total CDF
            and supply ``covariate`` (one normal score per member) instead.
        covariate : optional (member, site) array or (Y, X)/(member, Y, X) DataArray of
            z_S values, e.g. normal scores of bias-corrected dynamical-model totals
            (bypasses the tercile step entirely)
        target : Dataset/DataArray with Y, X (``bayesian=True`` only): simulate on
            another grid with GP-predicted coefficients
        calibrated_covariate : for a user ``covariate``, treat its members as samples
            of the target distribution and deconvolve them (default). False feeds
            the values unchanged (as in fitting).
        seasonal_temperature : in 'domain_three' mode, a mapping of TMIN and/or
            TMAX to seasonal means in degrees Celsius. Each value may be a
            scalar, an (member,) vector paired with precipitation members, or
            a (Y,X)/(member,Y,X) gridded DataArray. Grids are area averaged;
            omitted variables use their reference-period domain means.
            These are predictor covariates, not constraints guaranteeing the
            simulated temperature seasonal mean equals the supplied mean.
        Neither -> climatological covariate z_S ~ N(0, 1).
        """
        if not hasattr(self, "b_occ_"):
            raise RuntimeError("Call fit before generate")
        n_members, member_start = int(n_members), int(member_start)
        if n_members < 1 or member_start < 0:
            raise ValueError("n_members must be positive and member_start nonnegative")
        if seasonal_total is not None:
            if self.seasonal_covariates != "domain_three":
                raise ValueError("seasonal_total requires seasonal_covariates='domain_three'")
            if probabilities is not None or covariate is not None or domain_probabilities is not None:
                raise ValueError("seasonal_total cannot be combined with other precipitation forecasts")
            totals = np.broadcast_to(np.asarray(seasonal_total, float), (n_members,))
            if not np.isfinite(totals).all() or np.any(totals < 0):
                raise ValueError("seasonal_total requires finite nonnegative millimetres per member")
            covariate = self.totals_to_covariate(totals)[:, 0]
            calibrated_covariate = False
        if domain_probabilities is not None:
            if self.seasonal_covariates != "domain_three" or probabilities is not None or covariate is not None:
                raise ValueError("domain_probabilities requires domain_three and no other rain forecast")
            direct_p = np.asarray(domain_probabilities, float)
            if direct_p.shape != (3,) or not np.isfinite(direct_p).all() or np.any(direct_p < 0):
                raise ValueError("domain_probabilities must be finite nonnegative PB,PN,PA")
            if not np.isclose(direct_p.sum(), 1., atol=1e-6):
                raise ValueError("domain_probabilities must sum to one")
        if probabilities is not None and covariate is not None:
            raise ValueError("Supply probabilities or covariate, not both")
        dates = season_dates(int(year), self.months)
        if target is not None:
            ty, tx = np.asarray(target["Y"].values), np.asarray(target["X"].values)
            tlat, tlon = np.meshgrid(ty, tx, indexing="ij")
            lat, lon, yy, xx = tlat.ravel(), tlon.ravel(), ty, tx
            _, _, xyz_t = _coordinates(lat, lon)
            _, _, xyz_s = _coordinates(self.lat_, self.lon_)
            nearest = np.argmin(_chord(xyz_t, xyz_s), axis=1)
        else:
            lat, lon, yy, xx, nearest = self.lat_, self.lon_, self.y_, self.x_, np.arange(self.lat_.size)
        ns = lat.size
        temp_score, temp_physical = self._temperature_scores(seasonal_temperature, n_members, yy, xx)
        fields = {}
        seed = self.seed if _seed is None else int(_seed)

        def draw(step, stream, key, month_number=None):
            hook = getattr(self, "_posterior_spatial_draw", None)
            if hook is not None:
                result = hook(n_members, member_start, step, stream, key, month_number, lat, lon, seed)
                if result is not None:
                    return result
            if self.seasonal_covariates == "domain_three" and month_number is not None:
                monthly_key = f"{key}_month_{month_number}"
                if monthly_key in self.spatial_models_:
                    key, stream = monthly_key, stream + 1000 * int(month_number)
            if stream not in fields:
                if self.spatial == "independent" or key not in self.spatial_models_:
                    fields[stream] = IndependentField(lat, lon, seed=seed)
                else:
                    fields[stream] = GaussianSpatialField(lat, lon, model=self.spatial_models_[key],
                                                          seed=seed, n_features=self.n_features)
            return fields[stream].sample(n_members + member_start, step, stream)[member_start:]

        # seasonal covariate (target distribution), then deconvolution
        if covariate is not None:
            if isinstance(covariate, xr.DataArray) and {"Y", "X"}.issubset(covariate.dims):
                if (set(covariate.dims) - {"member", "Y", "X"}
                        or not np.array_equal(covariate.Y.values, yy)
                        or not np.array_equal(covariate.X.values, xx)):
                    raise ValueError("covariate Y,X coordinates must exactly match the simulation grid")
                dims = ("member", "Y", "X") if "member" in covariate.dims else ("Y", "X")
                covariate = covariate.transpose(*dims)
            z = np.asarray(covariate.values if isinstance(covariate, xr.DataArray) else covariate, float)
            if self.seasonal_covariates == "domain_three" and z.ndim == 1 and z.size == n_members:
                z = np.broadcast_to(z[:, None], (n_members, ns)).copy()
            else:
                z = np.broadcast_to(z.reshape((-1, ns)) if z.ndim > 1 else z[None], (n_members, ns)).copy()
            if self.seasonal_covariates == "domain_three" and not np.allclose(z, z[:, :1], equal_nan=True):
                raise ValueError("domain_three requires one domain-wide precipitation covariate per member")
            if calibrated_covariate:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean, variance = np.nanmean(z, 0), np.nanvar(z, 0)
                variance = np.where(np.isfinite(z).sum(0) > 2, variance, 1.)
                z_used = self._deconvolve(z, mean, variance, nearest)
            else:
                z_used = z
        else:
            eps = draw(-5, STREAM_SEASON, "glm_season")
            if domain_probabilities is not None:
                p = direct_p[:, None]
                eps = eps[:, :1]
                z = self.sample_covariate(p, eps)
                m, s2 = self._target_moments(p)
                z = np.broadcast_to(z, (n_members, ns))
                m, s2 = np.broadcast_to(m, (ns,)), np.broadcast_to(s2, (ns,))
            elif probabilities is not None:
                grid = xr.Dataset(coords={"Y": yy, "X": xx})
                p = prepare_probabilities(probabilities, target=grid).transpose("probability", "Y", "X").values.reshape(3, ns)
                if self.seasonal_covariates == "domain_three":
                    # This approximates a forecast of the domain total. The
                    # mean of local tercile probabilities does not identify
                    # the domain-total distribution or spatial dependence.
                    weight = np.maximum(np.cos(np.deg2rad(lat)), 0.)
                    support = (self.domain_support_ if np.array_equal(yy, self.y_)
                               and np.array_equal(xx, self.x_) else np.ones(ns, dtype=bool))
                    if not np.isfinite(p[:, support]).all():
                        raise ValueError("domain_three probability forecast must cover the fixed domain support")
                    if weight[support].sum() <= 0:
                        raise ValueError("domain_three requires positive cosine-latitude area weights")
                    p = np.average(p[:, support], weights=weight[support], axis=1)[:, None]
                    eps = eps[:, :1]
                z = self.sample_covariate(p, eps)
                m, s2 = self._target_moments(p)
            else:
                z, m, s2 = self._normal_to_rain_covariate(eps), np.zeros(ns), np.ones(ns)
                if self.seasonal_covariates == "domain_three":
                    z = np.broadcast_to(z[:, :1], (n_members, ns))
            if self.seasonal_covariates == "domain_three" and probabilities is not None:
                z = np.broadcast_to(z, (n_members, ns))
                m, s2 = np.broadcast_to(m, (ns,)), np.broadcast_to(s2, (ns,))
            z_used = self._deconvolve(z, m, s2, nearest)
        par = self._parameters(n_members, member_start, None if target is None else (lat, lon))
        H = self._harm(dates)
        nv = len(self.variables_)
        out = {"PRCP": np.full((n_members, len(dates), ns), np.nan)}
        for v in self.variables_:
            out[v] = np.full((n_members, len(dates), ns), np.nan)
        pw = np.nan_to_num(self.p_first_wet_[nearest], nan=0.5)
        prev = ndtr(draw(-3, STREAM_OCC, "glm_occurrence")) < pw[None]
        lagx = {v: np.broadcast_to(self.init_cont_[v][nearest], (n_members, ns)).copy() for v in self.variables_}
        trend = np.full((n_members, ns), (int(year) - self.year_center_) / self.year_scale_)
        chol = self.chol_cont_[nearest]
        k = np.exp(par["logk"][..., 0])
        for t in range(len(dates)):
            month_number = int(dates[t].month)
            h = np.broadcast_to(H[t], (n_members, ns, H.shape[1]))
            lag = prev.astype(float)
            eta = np.einsum("msp,msp->ms", self._x_occ(h, lag, z_used), np.broadcast_to(par["occ"], (n_members, ns, par["occ"].shape[-1])))
            wet = draw(t, STREAM_OCC, "glm_occurrence", month_number) < eta
            amount_eta = np.einsum("msp,msp->ms", self._x_amt(h, lag, z_used),
                                    np.broadcast_to(par["amt"], (n_members, ns, par["amt"].shape[-1])))
            if np.any(np.abs(amount_eta[np.isfinite(amount_eta)]) > 700):
                raise FloatingPointError("Gamma mean predictor exceeds floating-point exponential range")
            mu = np.exp(amount_eta)
            u = np.clip(ndtr(draw(t, STREAM_AMT, "glm_amount", month_number)), 1e-9, 1 - 1e-9)
            if self.amount_model == "raw_truncated":
                tail = gammaincc(k, k * self.thr / mu)
                from scipy.special import gammainccinv
                amount = gammainccinv(k, np.maximum((1 - u) * tail, np.finfo(float).tiny)) * mu / k
                amount = np.maximum(amount, np.nextafter(self.thr, np.inf))
            else:
                amount = self.thr + gammaincinv(np.broadcast_to(k, (n_members, ns)), u) * mu / k
            dry_value = 0.0
            if self.trace_rainfall:
                mth = dates[t].month - 1
                q, m_ = self.trace_p_[mth, nearest][None], self.trace_m_[mth, nearest][None]
                dry_value = np.where(u < q, np.minimum(2 * m_ * u / np.maximum(q, 1e-9), self.thr * (1 - 1e-6)), 0.)
            out["PRCP"][:, t] = np.where(wet, amount, dry_value)
            if nv:
                e = np.stack([draw(t, STREAM_CONT + i, f"glm_{v}", month_number)
                              for i, v in enumerate(self.variables_)], -1)
                monthly_chol = (chol if self.chol_cont_month_ is None
                                else self.chol_cont_month_[month_number - 1, nearest])
                e = np.einsum("sij,msj->msi", monthly_chol, e)
                lag_min = lagx.get("TMIN")
                lag_max = lagx.get("TMAX")
                for i, v in enumerate(self.variables_):
                    b = np.broadcast_to(par[f"c_{v}"], (n_members, ns, par[f"c_{v}"].shape[-1]))
                    if self.seasonal_covariates == "domain_three" and v in ("TMIN", "TMAX"):
                        xc = self._x_temperature(h, wet.astype(float), lag_min, lag_max,
                                                 temp_score["TMIN"], temp_score["TMAX"], trend)
                    else:
                        xc = self._x_cont(h, wet.astype(float), lagx[v], z_used)
                    mean = np.einsum("msp,msp->ms", xc, b)
                    sd = np.exp(np.where(wet, par[f"logsd_{v}"][..., 1], par[f"logsd_{v}"][..., 0]))
                    g = mean + sd * e[..., i]
                    # Both temperature equations use yesterday's TMIN/TMAX;
                    # commit both lag values only after fitting both equations.
                    if v in ("TMIN", "TMAX") and self.seasonal_covariates == "domain_three":
                        pass
                    else:
                        lagx[v] = g
                    out[v][:, t] = _inverse_transform(v, g)
                if self.seasonal_covariates == "domain_three":
                    lagx["TMIN"] = out["TMIN"][:, t]
                    lagx["TMAX"] = out["TMAX"][:, t]
            prev = wet
        for lo_, hi_ in (("TMIN", "TMAX"), ("HUMIN", "HUMAX")):
            if lo_ in out and hi_ in out:
                a, b = out[lo_], out[hi_]
                out[lo_], out[hi_] = np.minimum(a, b), np.maximum(a, b)
        active = np.isfinite(z_used)
        if target is None:
            active &= self.valid_[None]
        for v in out:
            out[v] = np.where(active[:, None], out[v], np.nan)
            if np.isinf(out[v]).any() or np.any(np.abs(out[v][np.isfinite(out[v])]) > np.finfo(np.float32).max):
                raise FloatingPointError(f"Generated {v} exceeds finite float32 output range; inspect forecast extrapolation")
        coords = {"member": np.arange(member_start, member_start + n_members), "T": dates, "Y": yy, "X": xx}
        ds = xr.Dataset({v: (("member", "T", "Y", "X"), a.reshape(n_members, len(dates), len(yy), len(xx)).astype("float32"))
                         for v, a in out.items()}, coords=coords)
        ds["season_zscore"] = (("member", "Y", "X"), z.reshape(n_members, len(yy), len(xx)).astype("float32"))
        ds["season_zscore"].attrs["description"] = (
            "Target member covariate: affine standardized physical domain total"
            if self.rainfall_covariate == "standardized_total" else
            "Target seasonal z_S of each member: forecast normal score of total")
        ds["covariate_used"] = (("member", "Y", "X"), np.asarray(z_used).reshape(n_members, len(yy), len(xx)).astype("float32"))
        ds["covariate_used"].attrs["description"] = "z_S value fed to the GLMs after transmission calibration"
        for v, physical in temp_physical.items():
            ds[f"seasonal_{v}_covariate"] = ("member", physical.astype("float32"))
            ds[f"seasonal_{v}_covariate"].attrs.update(
                units=self.units_.get(v, "degrees Celsius"),
                description="Domain-mean seasonal temperature forecast supplied to the GLM")
        for v in out:
            ds[v].attrs["units"] = "mm d-1" if v == "PRCP" else self.units_.get(v, "")
        ds.attrs.update(generator="was-disaggregation Verdin-type GLM" + (" + empirical-Bayes GP layer" if self.bayesian else ""),
                        total_sampling=self.total_sampling if covariate is None else "user covariate",
                        seasonal_covariates=self.seasonal_covariates,
                        season_months=",".join(map(str, self.months)), climatology=f"{self.climatology[0]}-{self.climatology[1]}",
                        leap_day_policy="February 29 excluded")
        ds.attrs.update(amount_model=self.amount_model, wet_rule=self.wet_rule,
                        rainfall_covariate=self.rainfall_covariate,
                        paper_protocol=self.paper_protocol or "none")
        if self.seasonal_covariates == "domain_three":
            ds.attrs["rain_forecast_aggregation"] = (
                "direct domain-tercile probabilities" if domain_probabilities is not None
                else "local probabilities area averaged (approximate)" if probabilities is not None
                else "physical domain-total ensemble supplied by user" if seasonal_total is not None
                else "domain-total covariates supplied by user" if covariate is not None
                else "climatological domain-total covariates")
        if seasonal_total is not None:
            ds["seasonal_PRCP_covariate"] = ("member", totals.astype("float32"))
            ds["seasonal_PRCP_covariate"].attrs.update(units="mm", description="Supplied domain seasonal rain total predictor")
        return ds

    def covariate_transmission(self, generated: xr.Dataset):
        """Slope/correlation between member covariate z_S and the normal score of
        the generated total (how well the forecast covariate reaches the output)."""
        tot = generated.PRCP.sum("T", skipna=False).values.reshape(generated.sizes["member"], -1)
        zc = generated.season_zscore.values.reshape(generated.sizes["member"], -1)
        ref = self.ref_totals_
        out = []
        for s in range(tot.shape[1]):
            r = ref[:, s][np.isfinite(ref[:, s])]
            if (r.size < 3 or not np.isfinite(tot[:, s]).all()
                    or not np.isfinite(zc[:, s]).all() or np.std(zc[:, s]) <= 1e-12
                    or tot.shape[0] < 3):
                continue
            zg = ndtri(np.clip((np.searchsorted(r, tot[:, s]) + 0.5) / (r.size + 1), 1e-4, 1 - 1e-4))
            out.append((np.polyfit(zc[:, s], zg, 1)[0], np.corrcoef(zc[:, s], zg)[0, 1]))
        if not out:
            return {"slope_mean": np.nan, "correlation_mean": np.nan}
        out = np.asarray(out)
        return {"slope_mean": float(out[:, 0].mean()), "correlation_mean": float(out[:, 1].mean())}


__all__ = ["GLMWeatherGenerator", "CoefficientGP", "fit_probit", "fit_gamma_log", "fit_linear"]
