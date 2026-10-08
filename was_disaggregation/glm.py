"""GLM weather generator with the seasonal forecast as a covariate (0.5.0).

Verdin et al. (2018, J. Hydrol.) condition a GLM weather generator (Furrer &
Katz 2007; Kleiber et al. 2012) on seasonal climate. The seasonal total enters
the daily models as a covariate, and seasonal totals are sampled from the
forecast. Verdin et al. (2019, BayGEN) make the coefficients spatial Gaussian
processes estimated in a Bayesian space-time hierarchy.

In contrast to "weight the years, then refit", the forecast is not a
reweighting of history. Every historical season contributes to the fit, and the
forecast only changes the covariate value at simulation time.

Model at each cell (probit occurrence, Gamma amounts, Gaussian other variables)::

    P(wet_t) = Phi(X_occ,t . b_occ),  X_occ = [1, harmonics(doy), wet_{t-1}, z_S, z_S * wet_{t-1}]
    rain_t - thr | wet ~ Gamma(k, mu_t / k),  log mu_t = X_amt,t . b_amt,  X_amt = [1, harmonics, wet_{t-1}, z_S]
    g(x_t) = X_c,t . b_c + sd_{wet/dry} * e_t,   X_c = [1, harmonics, wet_t, g(x_{t-1}), z_S]

``z_S`` is the normal score Phi^-1(F0(S)) of the season's total S, where F0 is
the reference-period climatology. Using the normal score instead of the raw
total makes the covariate scale-free. A tercile forecast then maps to a
distribution of z_S directly (``tercile_normal_forecast``, or sampling within
the forecast classes), and the covariate means the same at any location. This
is also what allows kriging to cells without data (BayGEN).

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
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize
from scipy.special import digamma, gammaincinv, ndtr, ndtri, polygamma

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
    sd = np.where(np.isfinite(sd), sd, np.sqrt(np.nanmean(resid ** 2, axis=1))[:, None])
    s2 = np.nanmean(resid ** 2, axis=1)
    cov = np.linalg.inv(XtX + ridge * np.eye(p)) * s2[:, None, None]
    n = np.maximum(m.sum(1), 2)
    return beta, cov, np.maximum(sd, 1e-6), resid, 1.0 / (2 * n)


# ---------------------------------------------------------------------------
# Spatial Gaussian-process smoothing of coefficients (empirical-Bayes BayGEN)
# ---------------------------------------------------------------------------
def _chord(xyz_a, xyz_b):
    return np.sqrt(np.maximum(((xyz_a[:, None, :] - xyz_b[None, :, :]) ** 2).sum(-1), 0))


class CoefficientGP:
    """Independent GP prior m + GP(0, tau^2 exp(-d/rho)) per coefficient.

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
    """Verdin et al. (2018) GLM generator; ``bayesian=True`` for a BayGEN-type GP layer.

    Parameters
    ----------
    months, climatology, wet_threshold : as in :class:`WeatherGenerator`
    harmonics : number of annual harmonics in every linear predictor
    interaction : include z_S x wet_{t-1} in the occurrence model
    total_sampling : 'normal' draws z_S from N(mu, sigma) matched to the tercile
        forecast; 'tercile' draws a class from PB/PN/PA and a uniform position
        inside it (sampling "seasonal totals from the forecast" as Verdin et al.)
    spatial : 'distance' (fitted kernels, Gaussian fields) or 'independent'
    bayesian : GP smoothing of all coefficients (empirical-Bayes BayGEN)
    parameter_uncertainty : in Bayesian mode, draw one coefficient field per member
    trace_rainfall : simulate sub-threshold rain on dry days (climatological)
    calibrate_covariate : deconvolve the covariate (see ``calibrate``). The
        historical z_S is the realised outcome: it contains the year's weather
        noise, which the daily simulation adds a second time. Without
        calibration the seasonal totals are over-dispersed (demo: SD 140 vs
        101 mm observed).
    """

    def __init__(self, months=(7, 8, 9), climatology=(1991, 2020), wet_threshold=1.0, harmonics=2,
                 interaction=True, total_sampling="normal", spatial="distance", bayesian=False,
                 parameter_uncertainty=True, n_hyper_sites=400, n_features=256, max_pairs=300,
                 max_spatial_sites=64, trace_rainfall=True, seed=42, chunk_sites=256,
                 calibrate_covariate=True, calibration_members=300):
        if total_sampling not in {"normal", "tercile"}:
            raise ValueError("total_sampling must be 'normal' or 'tercile'")
        if spatial not in {"distance", "independent"}:
            raise ValueError("spatial must be 'distance' or 'independent'")
        self.months, self.climatology = validate_months(months), tuple(climatology)
        self.thr, self.harmonics, self.interaction = float(wet_threshold), int(harmonics), bool(interaction)
        self.total_sampling, self.spatial, self.bayesian = total_sampling, spatial, bool(bayesian)
        self.parameter_uncertainty, self.n_hyper_sites = bool(parameter_uncertainty), int(n_hyper_sites)
        self.n_features, self.max_pairs, self.max_spatial_sites = int(n_features), int(max_pairs), int(max_spatial_sites)
        self.trace_rainfall, self.seed, self.chunk_sites = bool(trace_rainfall), int(seed), int(chunk_sites)
        self.calibrate_covariate, self.calibration_members = bool(calibrate_covariate), int(calibration_members)
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
        return np.concatenate([H, lag[..., None], z[..., None]], -1)

    def _x_cont(self, H, wet, lagx, z):
        return np.concatenate([H, wet[..., None], lagx[..., None], z[..., None]], -1)

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
        self.units_ = {v: obs[v].attrs.get("units", "") for v in obs.data_vars}
        # day before each season: lag covariates
        stacked = obs.stack(site=("Y", "X")).transpose("T", "site")
        pre_dates = [season_dates(int(y), self.months)[0] - pd.Timedelta(days=1) for y in self.years_]
        pre = stacked.reindex(T=pd.DatetimeIndex(pre_dates)).load()
        pre_rain = pre["PRCP"].values                                 # (year, site)
        # seasonal covariate: normal score of the season total
        totals = np.where(np.isfinite(rain).all(1), np.nansum(rain, 1), np.nan)
        self.z_hist_ = normal_scores(totals, self.years_, climatology=self.climatology)
        _, self.thresholds_ = classify_seasons(totals, self.years_, climatology=self.climatology)
        ref = (self.years_ >= self.climatology[0]) & (self.years_ <= self.climatology[1])
        self.ref_totals_ = np.sort(np.where(np.isfinite(totals[ref]), totals[ref], np.nan), axis=0)
        H = self._harm(season_dates(int(self.years_[0]), self.months))  # (nd, h)
        self.H_ = H
        wet = np.where(np.isfinite(rain), (rain >= self.thr).astype(float), np.nan)
        lag = np.concatenate([np.where(np.isfinite(pre_rain), (pre_rain >= self.thr).astype(float), np.nan)[:, None], wet[:, :-1]], 1)
        z = np.broadcast_to(self.z_hist_[:, None, :], rain.shape)
        # (site, N) layout
        flat = lambda a: np.moveaxis(a, -1, 0).reshape(ns, ny * nd)
        # Gregorian day-of-year differs after February in leap years. The
        # historical design must use each season's own dates, as generation does.
        Hn = np.stack([self._harm(season_dates(int(y), self.months))
                       for y in self.years_]).reshape(ny * nd, -1)
        Y_wet, L_wet, Z = flat(wet), flat(lag), flat(z)
        ok_occ = np.isfinite(Y_wet) & np.isfinite(L_wet) & np.isfinite(Z)
        excess = flat(rain) - self.thr
        ok_amt = ok_occ & (Y_wet == 1)
        p_occ = H.shape[1] + 2 + int(self.interaction)
        p_amt = H.shape[1] + 2
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
            b, c, k, vk = fit_gamma_log(Xa, np.maximum(np.nan_to_num(excess[sl]), 0.01), ok_amt[sl])
            self.b_amt_[sl], self.c_amt_[sl], self.shape_[sl], self.var_logk_[sl] = b, c, k, vk
        # other variables: transformed linear models with AR(1) and wet/dry effect
        self.b_cont_, self.c_cont_, self.sd_cont_, self.var_logsd_, resid = {}, {}, {}, {}, {}
        self.init_cont_ = {}
        for v in self.variables_:
            g = _transform(v, vals[v])
            gpre = _transform(v, pre[v].values)
            glag = np.concatenate([gpre[:, None], g[:, :-1]], 1)
            Gy, Gl = flat(g), flat(glag)
            ok = ok_occ & np.isfinite(Gy) & np.isfinite(Gl)
            p_c = H.shape[1] + 3
            B, C = np.zeros((ns, p_c)), np.zeros((ns, p_c, p_c))
            SD, R = np.ones((ns, 2)), np.full((ns, ny * nd), np.nan)
            VL = np.ones(ns)
            for s0 in range(0, ns, self.chunk_sites):
                sl = slice(s0, min(ns, s0 + self.chunk_sites))
                Hs = np.broadcast_to(Hn[None], (sl.stop - sl.start,) + Hn.shape)
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
        # trace rainfall (climatological, per month) and first-day wet probability
        mon = self.month_
        self.trace_p_ = np.zeros((12, ns)); self.trace_m_ = np.zeros((12, ns))
        for mth in np.unique(mon):
            x = rain[:, mon == mth]
            dry = np.isfinite(x) & (x < self.thr)
            tr = dry & (x > 0)
            self.trace_p_[mth - 1] = tr.sum((0, 1)) / np.maximum(dry.sum((0, 1)), 1)
            self.trace_m_[mth - 1] = np.where(tr, x, 0).sum((0, 1)) / np.maximum(tr.sum((0, 1)), 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pw = np.nanmean(np.where(np.isfinite(pre_rain), pre_rain >= self.thr, np.nan), axis=0)
            self.p_first_wet_ = np.where(np.isfinite(pw), pw, np.nanmean(wet[:, 0], axis=0))
        self.valid_ = np.isfinite(self.z_hist_).any(0) & np.isfinite(self.b_occ_).all(1)
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
        self.transmission_ = None
        if self.calibrate_covariate:
            self.calibrate(self.calibration_members)
        self.diagnostics_ = {
            "model": "Verdin et al. (2018) GLM: probit occurrence, Gamma(log) amounts, Gaussian AR(1) other variables",
            "covariate": "z_S = normal score of the season total (reference-period climatology)",
            "bayesian": ("empirical-Bayes GP on coefficients (BayGEN-type), hyperparameters "
                         + str([tuple(round(x, 4) for x in h) for h in self.gp_.hyper]) if self.bayesian else "off"),
            "occurrence_z_coefficient_mean": float(np.nanmean(self.b_occ_[:, self.H_.shape[1] + 1])),
            "amount_z_coefficient_mean": float(np.nanmean(self.b_amt_[:, -1])),
            "training_years": self.years_.tolist(),
        }
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

        Two simulations with a climatological covariate, z ~ N(0, 1) and
        z ~ N(0, 1/4), give the standardized generated total (reference mean and
        SD). From them:
        the mean response slope a and offset b (regression in the first run),
        and the variance response Var_out(s^2) = nu + A s^2 (two-point fit;
        nu is the weather-noise variance the daily simulation adds by itself).
        ``generate`` then feeds the deconvolved covariate with mean (m - b)/a and
        variance max(s^2 - nu, 0.05 s^2)/A, where m and s^2 are the target
        (forecast) mean and variance. The generated totals then reproduce the
        target mean and spread of the normal score. Under climatology this
        reproduces the observed interannual spread.
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
            sim = self.generate(int(self.years_.max()) + 1, n_members, covariate=z, calibrated_covariate=False,
                                _seed=self.seed + 7919 + k)
            runs.append((z, sim.PRCP.sum("T", skipna=False).values.reshape(n_members, ns)))
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
        """Normal scores z_S of seasonal totals (member, site) at the training cells.

        Use it to drive the GLM with a dynamical model: bias-correct its daily
        ensemble (``was_disaggregation.dynamical``), sum each member's season, convert
        the totals here, and pass them as ``generate(covariate=...)``. No tercile
        step is involved."""
        if isinstance(totals, xr.DataArray) and set(totals.dims) == {"member", "Y", "X"}:
            if not (np.array_equal(totals.Y.values, self.y_) and np.array_equal(totals.X.values, self.x_)):
                raise ValueError("totals Y,X coordinates must match the training grid")
            totals = totals.transpose("member", "Y", "X")
        t = np.asarray(totals.values if isinstance(totals, xr.DataArray) else totals, float)
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
        if self.total_sampling == "normal":
            mu, sigma = tercile_normal_forecast(p)
            return mu, sigma ** 2
        u = (np.arange(4000) + 0.5) / 4000
        zz = self.sample_covariate(p, ndtri(u)[:, None] * np.ones((1, p.shape[1])), raw=True)
        return zz.mean(0), zz.var(0)

    def sample_covariate(self, probabilities, eps, raw=False):
        """z_S (member, site) from tercile probabilities (3, site) and N(0,1) fields eps."""
        p = np.asarray(probabilities, float)
        if self.total_sampling == "normal":
            mu, sigma = tercile_normal_forecast(p)
            return mu[None] + sigma[None] * eps
        u = np.clip(ndtr(eps), 1e-9, 1 - 1e-9)
        c1, c2 = p[0][None], (p[0] + p[1])[None]
        cls = (u > c1).astype(int) + (u > c2).astype(int)
        lo = np.where(cls == 0, 0, np.where(cls == 1, c1, c2))
        width = np.where(cls == 0, p[0][None], np.where(cls == 1, p[1][None], p[2][None]))
        pos = np.clip((u - lo) / np.maximum(width, 1e-9), 1e-6, 1 - 1e-6)
        return ndtri((cls + pos) / 3.0)

    # ---- simulation ---------------------------------------------------------
    def generate(self, year, n_members=20, probabilities=None, covariate=None, member_start=0, target=None,
                 calibrated_covariate=True, _seed=None):
        """Simulate one season.

        probabilities : DataArray (probability, Y, X), tercile forecast of the total
        covariate : optional (member, site) array or (Y, X)/(member, Y, X) DataArray of
            z_S values, e.g. normal scores of bias-corrected dynamical-model totals
            (bypasses the tercile step entirely)
        target : Dataset/DataArray with Y, X (``bayesian=True`` only): simulate on
            another grid with GP-predicted coefficients
        calibrated_covariate : for a user ``covariate``, treat its members as samples
            of the target distribution and deconvolve them (default). False feeds
            the values unchanged (as in fitting).
        Neither -> climatological covariate z_S ~ N(0, 1).
        """
        if not hasattr(self, "b_occ_"):
            raise RuntimeError("Call fit before generate")
        n_members, member_start = int(n_members), int(member_start)
        if n_members < 1 or member_start < 0:
            raise ValueError("n_members must be positive and member_start nonnegative")
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
        fields = {}
        seed = self.seed if _seed is None else int(_seed)

        def draw(step, stream, key):
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
            z = np.broadcast_to(z.reshape((-1, ns)) if z.ndim > 1 else z[None], (n_members, ns)).copy()
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
            if probabilities is not None:
                grid = xr.Dataset(coords={"Y": yy, "X": xx})
                p = prepare_probabilities(probabilities, target=grid).transpose("probability", "Y", "X").values.reshape(3, ns)
                z = self.sample_covariate(p, eps)
                m, s2 = self._target_moments(p)
            else:
                z, m, s2 = eps, np.zeros(ns), np.ones(ns)
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
        chol = self.chol_cont_[nearest]
        k = np.exp(par["logk"][..., 0])
        for t in range(len(dates)):
            h = np.broadcast_to(H[t], (n_members, ns, H.shape[1]))
            lag = prev.astype(float)
            eta = np.einsum("msp,msp->ms", self._x_occ(h, lag, z_used), np.broadcast_to(par["occ"], (n_members, ns, par["occ"].shape[-1])))
            wet = draw(t, STREAM_OCC, "glm_occurrence") < eta
            mu = np.exp(np.clip(np.einsum("msp,msp->ms", self._x_amt(h, lag, z_used),
                                          np.broadcast_to(par["amt"], (n_members, ns, par["amt"].shape[-1]))), -10, 10))
            u = np.clip(ndtr(draw(t, STREAM_AMT, "glm_amount")), 1e-9, 1 - 1e-9)
            amount = self.thr + gammaincinv(np.broadcast_to(k, (n_members, ns)), u) * mu / k
            dry_value = 0.0
            if self.trace_rainfall:
                mth = dates[t].month - 1
                q, m_ = self.trace_p_[mth, nearest][None], self.trace_m_[mth, nearest][None]
                dry_value = np.where(u < q, np.minimum(2 * m_ * u / np.maximum(q, 1e-9), self.thr * (1 - 1e-6)), 0.)
            out["PRCP"][:, t] = np.where(wet, amount, dry_value)
            if nv:
                e = np.stack([draw(t, STREAM_CONT + i, f"glm_{v}") for i, v in enumerate(self.variables_)], -1)
                e = np.einsum("sij,msj->msi", chol, e)
                for i, v in enumerate(self.variables_):
                    b = np.broadcast_to(par[f"c_{v}"], (n_members, ns, par[f"c_{v}"].shape[-1]))
                    mean = np.einsum("msp,msp->ms", self._x_cont(h, wet.astype(float), lagx[v], z_used), b)
                    sd = np.exp(np.where(wet, par[f"logsd_{v}"][..., 1], par[f"logsd_{v}"][..., 0]))
                    g = mean + sd * e[..., i]
                    lagx[v] = g
                    out[v][:, t] = _inverse_transform(v, g)
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
        coords = {"member": np.arange(member_start, member_start + n_members), "T": dates, "Y": yy, "X": xx}
        ds = xr.Dataset({v: (("member", "T", "Y", "X"), a.reshape(n_members, len(dates), len(yy), len(xx)).astype("float32"))
                         for v, a in out.items()}, coords=coords)
        ds["season_zscore"] = (("member", "Y", "X"), z.reshape(n_members, len(yy), len(xx)).astype("float32"))
        ds["season_zscore"].attrs["description"] = "target seasonal z_S of each member (forecast normal score of the total)"
        ds["covariate_used"] = (("member", "Y", "X"), np.asarray(z_used).reshape(n_members, len(yy), len(xx)).astype("float32"))
        ds["covariate_used"].attrs["description"] = "z_S value fed to the GLMs after transmission calibration"
        for v in out:
            ds[v].attrs["units"] = "mm d-1" if v == "PRCP" else self.units_.get(v, "")
        ds.attrs.update(generator="was-disaggregation GLM (Verdin et al. 2018)" + (" + BayGEN-type GP layer" if self.bayesian else ""),
                        total_sampling=self.total_sampling if covariate is None else "user covariate",
                        season_months=",".join(map(str, self.months)), climatology=f"{self.climatology[0]}-{self.climatology[1]}",
                        leap_day_policy="February 29 excluded")
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
