"""Raw-data Bayesian GLM weather generator inspired by Verdin et al. (2019).

This is an explicitly specified extension, not a numerical reproduction of
BayGEN. Unlike :class:`GLMWeatherGenerator`'s empirical-Bayes option, likelihoods
are evaluated on daily observations, GP hyperparameters are sampled, and
monthly spatial temperature covariance uncertainty reaches predictive draws.
No optional probabilistic-programming dependency is required.

References
----------
* Andrew Verdin, Balaji Rajagopalan, William Kleiber, Guillermo Podesta and
  Federico Bert (2019), "BayGEN: A Bayesian Space-Time Stochastic Weather
  Generator". https://doi.org/10.1029/2017WR022473
  Inspiration: daily likelihoods and spatial GP coefficient hierarchies.
  Proper GP mean priors, log-positive shape/SD, shared family covariance,
  elliptical-slice updates and separately fitted rainfall copulas define a
  package extension, not the exact BayGEN hierarchy or its NUTS implementation.
* Andrew Verdin, Balaji Rajagopalan, William Kleiber, Guillermo Podesta and
  Federico Bert (2018), "A Conditional Stochastic Weather Generator for
  Seasonal to Multi-Decadal Simulations".
  https://doi.org/10.1016/j.jhydrol.2015.12.036
  Background: optional seasonal covariates inherited from the explicit GLM
  variants in glm.py, rather than an undocumented change to BayGEN.
* Iain Murray, Ryan Prescott Adams and David J. C. MacKay (2010), "Elliptical
  Slice Sampling". https://proceedings.mlr.press/v9/murray10a.html
  Core: invariant slice updates of Gaussian-prior latent blocks.
* Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter and Paul-Christian
  Burkner (2021), "Rank-Normalization, Folding, and Localization: An Improved
  R-hat for Assessing Convergence of MCMC". https://doi.org/10.1214/20-BA1221
  Core: rank-normalized/folded split R-hat and bulk effective sample size.
  These numerical diagnostics do not prove model adequacy or discover every
  posterior mode.

Scientific references attribute methods; software authorship is recorded
separately in package metadata and copyright notices. Published BayGEN
experiments and improved predictive skill have not been reproduced here.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import expit, gammaincc, gammaln, log_ndtr, ndtri
from scipy.stats import rankdata

from .data import canonicalize_observations, season_dates, seasonal_cube
from .glm import GLMWeatherGenerator, _chord
from .multivariate import _transform
from .spatial import _coordinates


def elliptical_slice(current, log_likelihood, rng, max_steps=1000):
    """Invariant elliptical-slice update for a standard-normal latent vector.

    The target is exp(log_likelihood(z)) N(z; 0, I). No tuning of the Gaussian
    prior's proposal scale is required (Murray, Adams & MacKay, 2010).
    Iain Murray, Ryan Prescott Adams and David J. C. MacKay, "Elliptical Slice
    Sampling", https://proceedings.mlr.press/v9/murray10a.html; invariant
    Gaussian-prior update core, separate from the weather-model specification.
    """
    current = np.asarray(current, float)
    value = float(log_likelihood(current))
    if not np.isfinite(value):
        raise ValueError("elliptical_slice requires a finite starting log likelihood")
    threshold = value + np.log(rng.uniform())
    direction = rng.standard_normal(current.shape)
    angle = rng.uniform(0., 2 * np.pi)
    lower, upper = angle - 2 * np.pi, angle
    for _ in range(max_steps):
        proposed = current * np.cos(angle) + direction * np.sin(angle)
        if float(log_likelihood(proposed)) >= threshold:
            return proposed
        if angle < 0:
            lower = angle
        else:
            upper = angle
        angle = rng.uniform(lower, upper)
    raise RuntimeError("elliptical-slice bracket exhausted; inspect numerical likelihood")


def _split_diagnostics(samples):
    """Split variance ratio and initial-positive-sequence ESS."""
    a = np.asarray(samples, float)
    chains, draws = a.shape[:2]
    half = draws // 2
    if chains < 2 or half < 4:
        shape = a.shape[2:]
        return np.full(shape, np.nan), np.full(shape, np.nan)
    z = np.concatenate((a[:, :half], a[:, -half:]), axis=0)
    z = z.reshape(2 * chains, half, -1)
    means = z.mean(axis=1)
    within = z.var(axis=1, ddof=1).mean(axis=0)
    # Detect identical traces before mean-subtraction roundoff is interpreted
    # as genuine within-chain variability.
    within = np.where(np.all(np.ptp(z, axis=1) == 0, axis=0), 0., within)
    between = half * means.var(axis=0, ddof=1)
    variance = (half - 1) / half * within + between / half
    rhat = np.sqrt(np.divide(variance, within, out=np.full_like(variance, np.nan), where=within > 0))
    rhat = np.where((within == 0) & (between > 0), np.inf, rhat)
    centered = z - means[:, None]
    fft = np.fft.rfft(centered, n=2 * half, axis=1)
    acov = np.fft.irfft(fft * np.conj(fft), n=2 * half, axis=1)[:, :half]
    acov /= np.arange(half, 0, -1)[None, :, None]
    rho = 1 - np.divide(within[None] - acov.mean(axis=0), variance[None],
                        out=np.ones((half, z.shape[-1])), where=variance[None] > 0)
    rho[0] = 1.
    total = np.zeros(z.shape[-1])
    active = np.ones(z.shape[-1], dtype=bool)
    previous = np.full(z.shape[-1], np.inf)
    for lag in range(0, half - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        active &= pair > 0
        pair = np.minimum(pair, previous)
        total += np.where(active, pair, 0.)
        previous = pair
    ess = np.minimum(2 * chains * half / np.maximum(-1 + 2 * total, 1.), 2 * chains * half)
    ess = np.where((within == 0) & (between == 0), np.nan, ess)
    return rhat.reshape(a.shape[2:]), ess.reshape(a.shape[2:])


def chain_diagnostics(samples):
    """Rank-normalized/folded split R-hat and bulk ESS (Vehtari et al., 2021).

    Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter and Paul-Christian
    Burkner (2021), DOI 10.1214/20-BA1221; full title in module References.

    Arrays have (chain, draw, ...). R-hat is the larger rank and folded-rank
    split statistic. Bulk ESS uses rank-normalized draws with Geyer's paired,
    monotone initial-positive sequence. Small R-hat is a necessary numerical
    check; it does not establish scientific adequacy or detect every mode.
    """
    a = np.asarray(samples, float)
    count = a.shape[0] * a.shape[1]
    flat = a.reshape(count, -1)
    ranked = ndtri((rankdata(flat, axis=0) - .375) / (count + .25)).reshape(a.shape)
    folded = np.abs(flat - np.median(flat, axis=0))
    folded_rank = ndtri((rankdata(folded, axis=0) - .375) / (count + .25)).reshape(a.shape)
    rank_rhat, ess = _split_diagnostics(ranked)
    folded_rhat, _ = _split_diagnostics(folded_rank)
    return np.fmax(rank_rhat, folded_rhat), ess


class BayesianGLMWeatherGenerator(GLMWeatherGenerator):
    """Bayesian daily-likelihood GLM with spatial coefficient GP hierarchy.

    Inspired by Andrew Verdin, Balaji Rajagopalan, William Kleiber, Guillermo
    Podesta and Federico Bert (2019), BayGEN, DOI 10.1029/2017WR022473.
    This is an explicitly specified package extension, not an exact article
    implementation; the full title and numerical references are listed in the
    module References. It uses daily-data posterior inference, unlike the
    Gaussian-measurement empirical-Bayes option in GLMWeatherGenerator.

    Bernoulli/probit occurrence and Gamma wet-day amount likelihoods are
    marginal products, as in BayGEN's computational occurrence/amount fit.
    Gaussian continuous-variable likelihoods use joint spatial errors with
    a distinct correlation range/nugget for each calendar month. Coefficient
    fields, log-Gamma shape and log residual SD share a sampled exponential GP
    hierarchy within each likelihood family. Priors are specified directly,
    and MLEs are used only to initialize sampling.

    Differences from BayGEN include zero-centered proper GP mean priors,
    log-positive shape/SD, a shared coefficient covariance within each family,
    and optional seasonal covariates inherited from the package. Occurrence
    and amount residual copulas remain estimated separately rather than jointly
    sampled. Dense inference is intended for station networks/small grids.
    """

    def __init__(self, *, draws=500, tune=500, chains=4, prior_mean_sd=20.,
                 prior_log_scale_sd=1., prior_log_range_sd=1., max_sites=64,
                 residual_covariance="monthly", coefficient_mean="linear", **kwargs):
        for name, value, minimum in (("draws", draws, 8), ("tune", tune, 0),
                                     ("chains", chains, 2), ("max_sites", max_sites, 1)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if residual_covariance not in {"monthly", "independent"}:
            raise ValueError("residual_covariance must be monthly/independent")
        if coefficient_mean not in {"constant", "linear"}:
            raise ValueError("coefficient_mean must be constant/linear")
        for name, value in (("prior_mean_sd", prior_mean_sd), ("prior_log_scale_sd", prior_log_scale_sd),
                            ("prior_log_range_sd", prior_log_range_sd)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if kwargs.get("bayesian", False):
            raise ValueError("BayesianGLMWeatherGenerator fits daily likelihoods; omit empirical bayesian=True")
        kwargs["bayesian"] = False
        kwargs.setdefault("calibrate_covariate", False)
        self.draws, self.tune, self.chains = int(draws), int(tune), int(chains)
        self.prior_mean_sd = float(prior_mean_sd)
        self.prior_log_scale_sd = float(prior_log_scale_sd)
        self.prior_log_range_sd = float(prior_log_range_sd)
        self.max_sites, self.residual_covariance = int(max_sites), residual_covariance
        self.coefficient_mean = coefficient_mean
        super().__init__(**kwargs)

    def fit(self, observations):
        """Sample the explicitly specified joint posterior from daily data."""
        obs = canonicalize_observations(observations)
        ns = obs.sizes["Y"] * obs.sizes["X"]
        if ns > self.max_sites:
            raise ValueError(f"Dense Bayesian inference supports <= {self.max_sites} cells; subset/regrid first")
        if self.calibrate_covariate:
            raise ValueError("Fit Bayesian model with calibrate_covariate=False; calibrate explicitly afterwards")
        super().fit(obs)
        if not self.valid_.all():
            raise ValueError("Bayesian inference requires observed data at every training cell; subset masked cells")
        cube = seasonal_cube(obs, months=self.months).load()
        ny, nd = cube.sizes["season_year"], cube.sizes["day"]
        values = {v: cube[v].transpose("season_year", "day", "Y", "X").values.reshape(ny, nd, ns)
                  for v in cube.data_vars}
        pre_dates = [season_dates(int(y), self.months)[0] - pd.Timedelta(days=1) for y in self.years_]
        pre = obs.stack(site=("Y", "X")).transpose("T", "site").reindex(T=pre_dates)
        rain = values["PRCP"]
        compare = np.greater if self.wet_rule == "gt" else np.greater_equal
        wet = np.where(np.isfinite(rain), compare(rain, self.thr), np.nan)
        pre_rain = pre.PRCP.values
        lag = np.concatenate((np.where(np.isfinite(pre_rain), compare(pre_rain, self.thr), np.nan)[:, None],
                              wet[:, :-1]), axis=1)
        flat = lambda v: np.moveaxis(v, -1, 0).reshape(ns, ny * nd)
        h = np.stack([self._harm(season_dates(int(y), self.months)) for y in self.years_]).reshape(ny * nd, -1)
        h = np.broadcast_to(h[None], (ns,) + h.shape)
        ywet, lagwet = flat(wet), flat(lag)
        z = flat(np.broadcast_to(self.z_hist_[:, None], rain.shape))
        ok = np.isfinite(ywet) & np.isfinite(lagwet) & np.isfinite(z)
        self.distance_ = _chord(_coordinates(self.lat_, self.lon_)[2], _coordinates(self.lat_, self.lon_)[2])
        self.range_prior_center_ = float(np.median(self.distance_[self.distance_ > 0])) if ns > 1 else 100.
        self.coordinate_center_ = np.array([self.lat_.mean(), self.lon_.mean()])
        self.coordinate_scale_ = np.maximum(np.std(np.c_[self.lat_, self.lon_], axis=0), 1.)
        self.mean_design_ = self._mean_design(self.lat_, self.lon_)
        self.family_data_ = {
            "occ": dict(X=self._x_occ(h, np.nan_to_num(lagwet), np.nan_to_num(z)),
                        y=np.nan_to_num(ywet), mask=ok, estimate=self.b_occ_, n_beta=self.b_occ_.shape[1]),
            "amt": dict(X=self._x_amt(h, np.nan_to_num(lagwet), np.nan_to_num(z)),
                        y=flat(rain) - (self.thr if self.amount_model == "excess" else 0.),
                        mask=ok & (ywet == 1), estimate=np.c_[self.b_amt_, np.log(self.shape_)],
                        n_beta=self.b_amt_.shape[1]),
        }
        temperature_lags = {}
        for name in self.variables_:
            transformed = _transform(name, values[name])
            temperature_lags[name] = flat(np.concatenate((_transform(name, pre[name].values)[:, None],
                                                         transformed[:, :-1]), axis=1))
        trend = flat(np.broadcast_to(((self.years_ - self.year_center_) / self.year_scale_)[:, None, None], rain.shape))
        month_flat = np.tile(self.month_, ny)
        for name in self.variables_:
            y = flat(_transform(name, values[name]))
            l = temperature_lags[name]
            mask = ok & np.isfinite(y) & np.isfinite(l)
            if self.seasonal_covariates == "domain_three" and name in {"TMIN", "TMAX"}:
                zmin, zmax = [flat(np.broadcast_to(self.temperature_z_hist_[v][:, None, None], rain.shape))
                              for v in ("TMIN", "TMAX")]
                mask &= np.isfinite(temperature_lags["TMIN"]) & np.isfinite(temperature_lags["TMAX"])
                x = self._x_temperature(h, np.nan_to_num(ywet), np.nan_to_num(temperature_lags["TMIN"]),
                                        np.nan_to_num(temperature_lags["TMAX"]), zmin, zmax, trend)
            else:
                x = self._x_cont(h, np.nan_to_num(ywet), np.nan_to_num(l), np.nan_to_num(z))
            groups = []
            for month in np.unique(month_flat):
                selector = np.flatnonzero(month_flat == month)
                patterns, inverse = np.unique(mask[:, selector].T, axis=0, return_inverse=True)
                for i, pattern in enumerate(patterns):
                    sites = np.flatnonzero(pattern)
                    if sites.size:
                        groups.append((int(month), sites, selector[inverse == i]))
            self.family_data_[f"c_{name}"] = dict(
                X=x, y=np.nan_to_num(y), mask=mask, wet=np.nan_to_num(ywet), groups=groups,
                months=np.unique(month_flat), estimate=np.c_[self.b_cont_[name], np.log(self.sd_cont_[name])],
                n_beta=self.b_cont_[name].shape[1])
        self.posterior_, self.posterior_diagnostics_ = {}, {}
        for index, (name, data) in enumerate(self.family_data_.items()):
            # Center/scale nonconstant predictors before placing priors. Use
            # one common scale across sites to preserve a smooth basis on new
            # grids. The likelihood is unchanged; priors describe effects per
            # predictor SD rather than depending on Celsius/mm conventions.
            count = max(data["mask"].sum(), 1)
            shared_center = np.where(data["mask"][..., None], data["X"], 0.).sum(axis=(0, 1)) / count
            shared_scale = np.sqrt(np.where(data["mask"][..., None], (data["X"] - shared_center) ** 2, 0.).sum(axis=(0, 1)) / count)
            center = np.broadcast_to(shared_center, (ns, len(shared_center))).copy()
            scale = np.broadcast_to(shared_scale, center.shape).copy()
            center[:, 0], scale[:, 0] = 0., 1.
            scale = np.where(scale > 1e-10, scale, 1.)
            p = data["n_beta"]
            data["center"], data["scale"] = center, scale
            data["X"] = (data["X"] - center[:, None]) / scale[:, None]
            estimate = data["estimate"].copy()
            estimate[:, 0] += np.sum(estimate[:, 1:p] * center[:, 1:], axis=1)
            estimate[:, :p] *= scale
            data["estimate"] = estimate
            self.posterior_[name] = self._sample_family(name, data, index)
            rh, ess = chain_diagnostics(self.posterior_[name]["theta"])
            hrh, hess = chain_diagnostics(self.posterior_[name]["hyper"])
            mrh, mess = chain_diagnostics(self.posterior_[name]["mean"])
            self.posterior_diagnostics_[name] = {
                "rhat_max": float(np.nanmax(rh)), "ess_min": float(np.nanmin(ess)),
                "hyper_rhat_max": float(np.nanmax(hrh)), "hyper_ess_min": float(np.nanmin(hess)),
                "mean_rhat_max": float(np.nanmax(mrh)), "mean_ess_min": float(np.nanmin(mess)),
            }
        # The fitted joint likelihoods for each continuous variable are
        # independent conditional on observed lags/occurrence (BayGEN Eq3/4).
        self.chol_cont_[:] = np.eye(max(len(self.variables_), 1))
        self.chol_cont_month_ = None
        self.diagnostics_.update(
            bayesian="Raw daily likelihood, GP coefficient/hyperparameter posterior MCMC",
            posterior_diagnostics=self.posterior_diagnostics_,
            sampler="elliptical slice + conjugate Gibbs + Metropolis; adaptation only during discarded warmup",
            article_fidelity="Explicit Bayesian GLM extension, not exact BayGEN (2019)")
        if any(max(d["rhat_max"], d["hyper_rhat_max"], d["mean_rhat_max"]) > 1.05 for d in self.posterior_diagnostics_.values()):
            warnings.warn("Bayesian chains have split R-hat > 1.05; increase tune/draws and inspect traces before use",
                          RuntimeWarning, stacklevel=2)
        return self

    def _coefficient_cholesky(self, logs):
        tau, length = np.exp(logs[:2])
        return np.linalg.cholesky(tau ** 2 * np.exp(-self.distance_ / length)
                                  + 1e-6 * np.eye(len(self.distance_)))

    def _mean_design(self, lat, lon):
        if self.coefficient_mean == "constant":
            return np.ones((len(lat), 1))
        standardized = (np.c_[lat, lon] - self.coordinate_center_) / self.coordinate_scale_
        return np.c_[np.ones(len(lat)), standardized]

    def _log_likelihood(self, name, data, theta, residual_hyper):
        p = data["n_beta"]
        eta = np.einsum("snp,sp->sn", data["X"], theta[:, :p])
        mask = data["mask"]
        if not np.isfinite(eta[mask]).all():
            return -np.inf
        if name == "occ":
            return float(np.where(mask, log_ndtr(np.where(data["y"] == 1, eta, -eta)), 0.).sum())
        if name == "amt":
            if np.any(np.abs(theta[:, -1]) > 700) or np.any(np.abs(eta[mask]) > 700):
                return -np.inf
            k = np.exp(theta[:, -1])[:, None]
            y = np.where(mask, data["y"], 1.)
            if np.any(y[mask] <= 0):
                return -np.inf
            with np.errstate(over="ignore", invalid="ignore"):
                ll = k * (np.log(k) - eta) - gammaln(k) + (k - 1) * np.log(y) - k * y * np.exp(-eta)
            if not np.isfinite(ll[mask]).all():
                return -np.inf
            if self.amount_model == "raw_truncated":
                survival = gammaincc(k, k * self.thr * np.exp(-eta))
                if np.any(survival[mask] <= 0):
                    return -np.inf
                ll -= np.log(np.maximum(survival, np.finfo(float).tiny))
            return float(np.where(mask, ll, 0.).sum())
        logsd = np.where(data["wet"] == 1, theta[:, -1, None], theta[:, -2, None])
        if np.any(np.abs(logsd[mask]) > 30):
            return -np.inf
        residual = (data["y"] - eta) * np.exp(-logsd)
        result = -float(logsd[mask].sum())
        for month, sites, times in data["groups"]:
            if self.residual_covariance == "independent":
                result -= .5 * np.sum(residual[np.ix_(sites, times)] ** 2)
                result -= .5 * sites.size * times.size * np.log(2 * np.pi)
                continue
            j = int(np.searchsorted(data["months"], month))
            length, nugget = np.exp(residual_hyper[2 * j]), expit(residual_hyper[2 * j + 1])
            covariance = ((1 - nugget) * np.exp(-self.distance_[np.ix_(sites, sites)] / length)
                          + nugget * np.eye(sites.size))
            try:
                chol = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError:
                return -np.inf
            normalized = solve_triangular(chol, residual[np.ix_(sites, times)], lower=True, check_finite=False)
            result -= .5 * np.sum(normalized ** 2) + times.size * np.log(np.diag(chol)).sum()
            result -= .5 * sites.size * times.size * np.log(2 * np.pi)
        return result

    def _sample_family(self, name, data, index):
        ns, width = data["estimate"].shape
        n_resid = 2 * len(data.get("months", [])) if self.residual_covariance == "monthly" else 0
        reference = np.r_[0., np.log(self.range_prior_center_),
                           np.tile([np.log(self.range_prior_center_), -2.197224577], n_resid // 2)]
        prior_sd = np.r_[self.prior_log_scale_sd, self.prior_log_range_sd,
                          np.tile([self.prior_log_range_sd, 1.], n_resid // 2)]
        theta_draws = np.empty((self.chains, self.draws, ns, width))
        hyper_draws = np.empty((self.chains, self.draws, len(reference)))
        q = self.mean_design_.shape[1]
        means_draws = np.empty((self.chains, self.draws, q, width))
        accepted = []
        for chain in range(self.chains):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 91819, index, chain]))
            mean = np.linalg.lstsq(self.mean_design_, data["estimate"], rcond=None)[0]
            mean += rng.normal(0, .025, mean.shape)
            hyper = reference.copy()
            chol = self._coefficient_cholesky(hyper)
            white = solve_triangular(chol, data["estimate"] - self.mean_design_ @ mean, lower=True)
            white += rng.normal(0, .01, white.shape)
            theta = self.mean_design_ @ mean + chol @ white
            ll = self._log_likelihood(name, data, theta, hyper[2:])
            if not np.isfinite(ll):
                raise ValueError(f"Nonfinite starting likelihood for {name}; check data and fitted initialization")
            step_hyper = .08
            ac_hyper = 0
            for iteration in range(self.tune + self.draws):
                for column in rng.permutation(width):
                    def conditional_likelihood(candidate):
                        proposal_white = white.copy()
                        proposal_white[:, column] = candidate
                        return self._log_likelihood(name, data, self.mean_design_ @ mean + chol @ proposal_white, hyper[2:])
                    white[:, column] = elliptical_slice(white[:, column], conditional_likelihood, rng)
                theta = self.mean_design_ @ mean + chol @ white
                ll = self._log_likelihood(name, data, theta, hyper[2:])
                # Exact conjugate mean update with theta fixed. Alternating
                # centered and noncentered updates greatly reduces otherwise
                # severe mean/GP-scale confounding.
                design = solve_triangular(chol, self.mean_design_, lower=True)
                whitened_theta = solve_triangular(chol, theta, lower=True)
                mean_var = np.linalg.inv(design.T @ design + np.eye(q) / self.prior_mean_sd ** 2)
                mean = mean_var @ design.T @ whitened_theta + np.linalg.cholesky(mean_var) @ rng.standard_normal((q, width))
                white = solve_triangular(chol, theta - self.mean_design_ @ mean, lower=True)
                proposal_h = hyper + rng.normal(0, step_hyper, hyper.shape)
                # Far-tail exponent overflow is assigned zero posterior mass
                # in floating-point arithmetic, not silently clipped predictors.
                if np.any(np.abs(proposal_h) > 50):
                    hyper_accept = False
                else:
                    try:
                        new_chol = self._coefficient_cholesky(proposal_h)
                    except (np.linalg.LinAlgError, FloatingPointError):
                        new_chol = None
                    if new_chol is None:
                        # Numerically singular GP proposals have zero usable
                        # floating-point density; reject, preserve valid state.
                        if iteration >= self.tune:
                            draw = iteration - self.tune
                            theta_draws[chain, draw], hyper_draws[chain, draw], means_draws[chain, draw] = theta, hyper, mean
                        continue
                    new_white = solve_triangular(new_chol, theta - self.mean_design_ @ mean, lower=True)
                    new_ll = self._log_likelihood(name, data, theta, proposal_h[2:])
                    ratio = (new_ll - ll - width * (np.log(np.diag(new_chol)).sum() - np.log(np.diag(chol)).sum())
                             - .5 * (np.sum(new_white ** 2) - np.sum(white ** 2))
                             - .5 * np.sum(((proposal_h - reference) / prior_sd) ** 2)
                             + .5 * np.sum(((hyper - reference) / prior_sd) ** 2))
                    hyper_accept = np.log(rng.uniform()) < ratio
                    if hyper_accept:
                        hyper, chol, white, ll = proposal_h, new_chol, new_white, new_ll
                        ac_hyper += 1
                if iteration < self.tune:
                    rate = min(.05, 1 / np.sqrt(iteration + 1))
                    step_hyper *= np.exp(rate * (float(hyper_accept) - .25))
                else:
                    draw = iteration - self.tune
                    theta_draws[chain, draw], hyper_draws[chain, draw], means_draws[chain, draw] = theta, hyper, mean
            accepted.append(ac_hyper / (self.tune + self.draws))
        return dict(theta=theta_draws, hyper=hyper_draws, mean=means_draws, acceptance=np.asarray(accepted))

    def posterior_summary(self, probability=.95):
        """Credible intervals/R-hat/ESS in the fitted standardized predictor basis."""
        if not hasattr(self, "posterior_"):
            raise RuntimeError("Call fit before posterior_summary")
        if not 0 < probability < 1:
            raise ValueError("probability must lie strictly between zero and one")
        rows = []
        for name, posterior in self.posterior_.items():
            a = posterior["theta"]
            lower, upper = np.quantile(a, [(1 - probability) / 2, (1 + probability) / 2], axis=(0, 1))
            rhat, ess = chain_diagnostics(a)
            for site in range(a.shape[2]):
                for column in range(a.shape[3]):
                    rows.append(dict(family=name, site=site, parameter=column, mean=a[:, :, site, column].mean(),
                                     lower=lower[site, column], upper=upper[site, column],
                                     rhat=rhat[site, column], ess=ess[site, column]))
        return pd.DataFrame(rows)

    def _parameters(self, n_members, member_start, target_latlon):
        out, self._predictive_indices_ = {}, {}
        for index, (name, posterior) in enumerate(self.posterior_.items()):
            width = posterior["theta"].shape[-1]
            ns = len(self.lat_) if target_latlon is None else len(target_latlon[0])
            if ns > self.max_sites:
                raise ValueError(f"Dense GP prediction supports <= {self.max_sites} target cells; subset/regrid first")
            theta = np.empty((n_members, ns, width))
            selected = []
            for i, member in enumerate(range(member_start, member_start + n_members)):
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, 99191, index, member]))
                chain, draw = (rng.integers(self.chains), rng.integers(self.draws)) if self.parameter_uncertainty else (0, 0)
                selected.append((chain, draw))
                sample = posterior["theta"][chain, draw]
                if not self.parameter_uncertainty:
                    sample = posterior["theta"].mean(axis=(0, 1))
                if target_latlon is None:
                    theta[i] = sample
                else:
                    xyz = _coordinates(*target_latlon)[2]
                    training = _coordinates(self.lat_, self.lon_)[2]
                    ds = _chord(xyz, training)
                    dss = _chord(xyz, xyz)
                    hyper = (posterior["hyper"][chain, draw] if self.parameter_uncertainty
                             else posterior["hyper"].mean(axis=(0, 1)))
                    mean = (posterior["mean"][chain, draw] if self.parameter_uncertainty
                            else posterior["mean"].mean(axis=(0, 1)))
                    tau, length = np.exp(hyper[:2])
                    covariance = tau ** 2 * np.exp(-self.distance_ / length) + 1e-6 * np.eye(len(training))
                    cross = tau ** 2 * np.exp(-ds / length)
                    solved = np.linalg.solve(covariance, cross.T)
                    expected = self._mean_design(*target_latlon) @ mean + cross @ np.linalg.solve(
                        covariance, sample - self.mean_design_ @ mean)
                    conditional = tau ** 2 * np.exp(-dss / length) + 1e-6 * np.eye(ns) - cross @ solved
                    conditional = (conditional + conditional.T) / 2
                    values, vectors = np.linalg.eigh(conditional)
                    tolerance = 1e-8 * max(float(np.diag(conditional).max()), float(tau ** 2), 1.)
                    if values.min() < -tolerance:
                        raise np.linalg.LinAlgError("Predictive GP covariance has substantive negative eigenvalues")
                    theta[i] = (expected + (vectors * np.sqrt(np.maximum(values, 0))) @ rng.standard_normal((ns, width))
                                if self.parameter_uncertainty else expected)
                    exact = np.isclose(ds.min(axis=1), 0., atol=1e-8)
                    theta[i, exact] = sample[ds.argmin(axis=1)[exact]]
            self._predictive_indices_[name] = selected
            p = self.family_data_[name]["n_beta"]
            data = self.family_data_[name]
            if target_latlon is None:
                nearest = np.arange(len(self.lat_))
            else:
                nearest = _chord(_coordinates(*target_latlon)[2], _coordinates(self.lat_, self.lon_)[2]).argmin(axis=1)
            # Reparameterize back into the GLM's original predictor basis.
            theta[..., :p] /= data["scale"][nearest][None]
            theta[..., 0] -= np.sum(theta[..., 1:p] * data["center"][nearest, 1:][None], axis=-1)
            if name == "amt":
                out["amt"], out["logk"] = theta[..., :p], theta[..., p:p + 1]
            elif name.startswith("c_"):
                out[name], out[f"logsd_{name[2:]}"] = theta[..., :p], theta[..., p:]
            else:
                out[name] = theta
        return out

    def _posterior_spatial_draw(self, n, start, step, stream, key, month, lat, lon, seed):
        if not hasattr(self, "_predictive_indices_") or month is None or self.residual_covariance == "independent":
            return None
        name = f"c_{key[4:]}" if key.startswith("glm_") else ""
        if name not in self.posterior_:
            return None
        xyz = _coordinates(lat, lon)[2]
        distance = _chord(xyz, xyz)
        months = self.family_data_[name]["months"]
        j = int(np.searchsorted(months, month))
        result = np.empty((n, len(lat)))
        for i, (chain, draw) in enumerate(self._predictive_indices_[name]):
            logs = (self.posterior_[name]["hyper"][chain, draw, 2:] if self.parameter_uncertainty
                    else self.posterior_[name]["hyper"].mean(axis=(0, 1))[2:])
            length, nugget = np.exp(logs[2 * j]), expit(logs[2 * j + 1])
            covariance = (1 - nugget) * np.exp(-distance / length) + nugget * np.eye(len(lat))
            rng = np.random.default_rng(np.random.SeedSequence([seed, 71621, stream, step, start + i]))
            result[i] = np.linalg.cholesky(covariance) @ rng.standard_normal(len(lat))
        return result

    def generate(self, *args, **kwargs):
        if not hasattr(self, "posterior_"):
            raise RuntimeError("Call fit before generate")
        ds = super().generate(*args, **kwargs)
        ds.attrs["generator"] = "was-disaggregation Bayesian daily-likelihood GLM extension"
        ds.attrs["posterior_draws_per_chain"] = self.draws
        ds.attrs["posterior_chains"] = self.chains
        ds.attrs["residual_covariance"] = self.residual_covariance
        ds.attrs["article_fidelity"] = "Inspired by BayGEN; modified hierarchy explicitly documented"
        return ds


__all__ = ["BayesianGLMWeatherGenerator", "elliptical_slice", "chain_diagnostics"]
