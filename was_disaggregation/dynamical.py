"""Downscaling the dynamical forecast directly: no tercile step (0.5.0).

The tercile forecast discards what the dynamical model says about the daily
sequence: onset, dry spells, intraseasonal variability. This module works from
the model's daily ensemble instead (e.g. C3S / SEAS5 daily precipitation):

1. **Daily bias correction** (``DailyBiasCorrector``): per cell and calendar
   month, pooled over hindcast years and members,
   * ``"qm"``: empirical quantile mapping (Panofsky & Brier; Gudmundsson et al. 2012),
   * ``"loci_qm"``: local intensity scaling of the wet-day threshold (Schmidli
     et al. 2006), then quantile mapping of wet-day amounts,
   * any external object with ``fit``/``transform`` (e.g. ``WAS_MC_QM`` or
     ``WAS_MC_LOCI`` from ``was_markov_chain_bc``) via ``ExternalCorrector``.
2. **Non-homogeneous hidden Markov model** (``NHMM``; Hughes & Guttorp 1994;
   Robertson et al. 2004). Hidden daily weather states, with transitions
   driven by model predictors, emit multi-site occurrence and amounts. A
   stochastic downscaling route from daily model predictors. ``NHSMM`` (0.6.0)
   is the hidden semi-Markov version: explicit, predictor-dependent state
   durations, so long dry regimes are not cut short by geometric dwell times.
3. **Space-time dependence restoration** of calibrated marginals:
   * ``ensemble_copula_coupling`` (ECC-Q / ECC-R; Schefzik et al. 2013): the
     raw ensemble's rank structure,
   * ``schaake_shuffle`` with random historical dates (Clark et al. 2004b), or
     with **preferentially selected** dates whose observed trajectories resemble
     the forecast (``preferential_dates``, after Scheuerer et al. 2017).
4. ``DynamicalDownscaler`` chains them: correct, then pool a +/- window of
   days into calibrated marginals, then couple.

``synthetic_model_ensemble`` builds an ARTIFICIAL biased "dynamical model"
ensemble from observations. It exists only for demonstrations and tests, when
no C3S files are at hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize
from scipy.special import digamma, gammainc, gammaincinv, gammaln, logsumexp, polygamma

from .data import (season_dates, validate_months, canonicalize_observations,
                   _validate_grid, _convert_units)
from .nonparametric import schaake_shuffle
from .spatial import GaussianSpatialField, fit_distance_model
from scipy.special import ndtr


# ---------------------------------------------------------------------------
# helpers: seasons of a daily (member, T, Y, X) array
# ---------------------------------------------------------------------------
def season_blocks(da: xr.DataArray, years, months):
    """Complete daily seasons -> (year, member, day, site); omit February 29.

    Individual missing values remain NaN. Missing dates, duplicate daily
    timestamps and unsupported calendars are rejected rather than imputed.
    """
    months = validate_months(months)
    _validate_grid(da, time=True)
    years = list(years)
    if not years or len(set(years)) != len(years):
        raise ValueError("years must be a nonempty sequence of unique season years")
    if set(da.dims) - {"member", "T", "Y", "X"}:
        raise ValueError("daily arrays must have only member,T,Y,X dimensions")
    if "member" not in da.dims:
        da = da.expand_dims(member=[0])
    da = da.transpose("member", "T", "Y", "X")
    t = pd.DatetimeIndex(da["T"].values).normalize()
    da = da.assign_coords(T=t)
    out = []
    for y in years:
        d = season_dates(int(y), months)
        if not d.isin(t).all():
            raise ValueError(f"Season {y} has missing daily dates")
        out.append(da.sel(T=d).values.reshape(da.sizes["member"], len(d), -1))
    return np.stack(out)


def _padded_dates(year, months, window):
    """Padding in the same 365-day calendar as seasonal generation."""
    core = season_dates(int(year), months)
    before = pd.date_range(end=core[0] - pd.Timedelta(days=1), periods=window + 1)
    after = pd.date_range(start=core[-1] + pd.Timedelta(days=1), periods=window + 1)
    before = before[~((before.month == 2) & (before.day == 29))]
    after = after[~((after.month == 2) & (after.day == 29))]
    return before[-window:].append(core).append(after[:window]) if window else core


def _to_dataarray(arr, dates, y, x, name, members=None):
    """(member, day, site) -> DataArray (member, T, Y, X)."""
    m = arr.shape[0]
    return xr.DataArray(arr.reshape(m, len(dates), len(y), len(x)), dims=("member", "T", "Y", "X"),
                        coords={"member": np.arange(m) if members is None else members, "T": dates, "Y": y, "X": x},
                        name=name)


# ---------------------------------------------------------------------------
# 1. Daily bias correction
# ---------------------------------------------------------------------------
class DailyBiasCorrector:
    """Per-cell, per-calendar-month empirical correction of daily precipitation.

    ``method="qm"``: F_obs^-1(F_mod(x)) on all days, using ``n_quantiles``
    probability levels with linear interpolation. The top level is extrapolated
    with the constant difference of the highest quantiles.
    ``method="loci_qm"``: the model threshold t_m is set so that the model
    wet-day frequency (x > t_m) approximates the observed frequency (x >= wet_threshold);
    model days <= t_m become dry (0), and wet-day amounts are quantile-mapped
    onto observed wet-day amounts. Exact frequencies cannot generally be
    obtained with a deterministic threshold when model values are tied.
    """

    def __init__(self, method="loci_qm", wet_threshold=1.0, n_quantiles=200):
        if method not in {"qm", "loci_qm"}:
            raise ValueError("method must be 'qm' or 'loci_qm'")
        self.method, self.thr, self.nq = method, float(wet_threshold), int(n_quantiles)
        if not np.isfinite(self.thr) or self.thr <= 0 or self.nq < 2:
            raise ValueError("wet_threshold must be positive and finite; n_quantiles must be >= 2")

    def fit(self, model, obs, month):
        """model (year, member, day, site); obs (year, day, site); month (day,)."""
        model, obs, month = np.asarray(model, float), np.asarray(obs, float), np.asarray(month)
        if (model.ndim != 4 or obs.shape != (model.shape[0], model.shape[2], model.shape[3])
                or month.shape != (model.shape[2],) or 0 in model.shape):
            raise ValueError("model must be (year,member,day,site), obs (year,day,site), month (day,)")
        if not np.isin(month, np.arange(1, 13)).all():
            raise ValueError("month values must be calendar months 1..12")
        if (np.isfinite(model) & (model < 0)).any() or (np.isfinite(obs) & (obs < 0)).any():
            raise ValueError("precipitation must be nonnegative or missing")
        self.months_ = np.unique(month)
        ns = model.shape[-1]
        q = (np.arange(self.nq) + 0.5) / self.nq
        self.q_ = q
        self.mod_q_, self.obs_q_ = {}, {}
        self.mod_thr_ = {}
        self.valid_ = {}
        for m in self.months_:
            sel = month == m
            mod = np.moveaxis(model[:, :, sel, :], -1, 0).reshape(ns, -1)
            ob = np.moveaxis(obs[:, sel, :], -1, 0).reshape(ns, -1)
            self.valid_[m] = np.isfinite(mod).any(1) & np.isfinite(ob).any(1)
            if self.method == "qm":
                good = self.valid_[m]
                mq, oq = np.full((ns, self.nq), np.nan), np.full((ns, self.nq), np.nan)
                if good.any():
                    mq[good] = np.nanquantile(mod[good], q, axis=1).T
                    oq[good] = np.nanquantile(ob[good], q, axis=1).T
                self.mod_q_[m], self.obs_q_[m] = mq, oq
            else:
                f_obs = (np.isfinite(ob) & (ob >= self.thr)).sum(1) / np.maximum(np.isfinite(ob).sum(1), 1)
                t_mod = np.array([np.nanquantile(mod[s], 1 - f_obs[s])
                                  if self.valid_[m][s] and f_obs[s] > 0 else np.inf
                                  for s in range(ns)])
                for s in np.flatnonzero(self.valid_[m] & (f_obs == 1)):
                    t_mod[s] = np.nextafter(np.nanmin(mod[s]), -np.inf)
                self.mod_thr_[m] = t_mod
                mq, oq = np.full((ns, self.nq), np.nan), np.full((ns, self.nq), np.nan)
                for s in range(ns):
                    mw = mod[s][np.isfinite(mod[s]) & (mod[s] > t_mod[s])]
                    ow = ob[s][np.isfinite(ob[s]) & (ob[s] >= self.thr)]
                    if mw.size >= 5 and ow.size >= 5:
                        mq[s], oq[s] = np.quantile(mw, q), np.quantile(ow, q)
                self.mod_q_[m], self.obs_q_[m] = mq, oq
        return self

    def _map(self, x, mq, oq):
        out = np.full_like(x, np.nan)
        for s in range(x.shape[-1]):
            if not np.isfinite(mq[s]).all():
                continue
            xs = x[..., s]
            y = np.interp(xs, mq[s], oq[s])
            hi = xs > mq[s][-1]
            y = np.where(hi, oq[s][-1] + (xs - mq[s][-1]), y)        # constant-offset extrapolation
            out[..., s] = np.where(np.isfinite(xs), y, np.nan)
        return out

    def transform(self, model, month):
        """model (..., day, site) -> corrected, same shape."""
        if not hasattr(self, "months_"):
            raise RuntimeError("Call fit before transform")
        model, month = np.asarray(model, float), np.asarray(month)
        ns = next(iter(self.mod_q_.values())).shape[0]
        if model.ndim < 2 or model.shape[-1] != ns or month.shape != (model.shape[-2],):
            raise ValueError("model (...,day,site) and month (day,) must match the fitted sites")
        if not np.isin(month, self.months_).all():
            raise ValueError("forecast includes calendar months absent from calibration")
        if (np.isfinite(model) & (model < 0)).any():
            raise ValueError("precipitation must be nonnegative or missing")
        out = np.full(model.shape, np.nan)
        for m in self.months_:
            sel = month == m
            x = model[..., sel, :]
            if self.method == "qm":
                out[..., sel, :] = np.maximum(self._map(x, self.mod_q_[m], self.obs_q_[m]), 0)
            else:
                wet = x > self.mod_thr_[m]
                y = self._map(x, self.mod_q_[m], self.obs_q_[m])
                out[..., sel, :] = np.where(np.isfinite(x), np.where(wet, np.maximum(y, self.thr), 0.0), np.nan)
            out[..., sel, :] = np.where(self.valid_[m], out[..., sel, :], np.nan)
        return out


class ExternalCorrector:
    """Adapter for any corrector with ``fit`` / ``transform`` methods.

    Use it to plug in ``WAS_MC_QM`` / ``WAS_MC_LOCI`` from ``was_markov_chain_bc``
    (or any other package). ``fit_args`` builds the positional arguments from
    (model, obs, month), and ``transform_args`` from (model, month): adapt these
    two lambdas to the external signatures. Default: fit(model, obs), transform(model).
    """

    def __init__(self, corrector, fit_args=None, transform_args=None, fit_method="fit", transform_method="transform"):
        self.c = corrector
        self.fit_args = fit_args or (lambda model, obs, month: (model, obs))
        self.transform_args = transform_args or (lambda model, month: (model,))
        self.fit_method, self.transform_method = fit_method, transform_method

    def fit(self, model, obs, month):
        getattr(self.c, self.fit_method)(*self.fit_args(model, obs, month))
        return self

    def transform(self, model, month):
        return np.asarray(getattr(self.c, self.transform_method)(*self.transform_args(model, month)))


# ---------------------------------------------------------------------------
# 2. Non-homogeneous hidden Markov model
# ---------------------------------------------------------------------------
class NHMM:
    """Multi-site NHMM for daily rainfall (Hughes & Guttorp 1994; Robertson et al. 2004).

    Hidden states S_t in {0..K-1}. Transition probabilities
    P(S_t = j | S_{t-1} = i, x_t) ∝ exp(A_ij + lambda_j . x_t), with state 0 as
    reference (A_i0 = 0, lambda_0 = 0) and x_t the daily predictors (e.g.
    standardized ensemble-mean model rainfall). Given the state, sites are
    independent: wet with probability p_ks, and wet-day excess over the
    threshold is Gamma(k_ks, theta_ks * exp(g_k x_ta)). The amount predictor
    term (``amount_predictor`` = index a of the predictor; default the last one,
    the seasonal anomaly of ``nhmm_predictors``; None disables it) lets intensity follow
    the model as well as occurrence. That matters where interannual variance
    comes from intensity more than from wet-day frequency, as with AgERA5 over
    the Sudanian zone. Fitted by EM (Baum–Welch) with a
    numerical M-step for (A, lambda). Large-scale spatial coherence comes from
    the shared state; within-state spatial dependence is not modelled, which is
    the classical conditional-independence assumption.
    """

    def __init__(self, n_states=4, amounts=True, amount_predictor=-1, wet_threshold=1.0, n_iter=60, tol=1e-4, seed=0):
        self.K, self.amounts, self.thr = int(n_states), bool(amounts), float(wet_threshold)
        # index of the predictor scaling wet-day amounts (None: no amount predictor)
        self.amount_predictor = amount_predictor
        self._ai = 0 if amount_predictor is None else int(amount_predictor)
        self.n_iter, self.tol, self.seed = int(n_iter), float(tol), int(seed)
        if self.K < 2 or self.n_iter < 1 or self.tol < 0 or not np.isfinite(self.thr) or self.thr <= 0:
            raise ValueError("n_states must be >= 2, n_iter positive, tol nonnegative and wet_threshold positive")

    def _validated_inputs(self, rain, predictors):
        rain, x = np.asarray(rain, float), np.asarray(predictors, float)
        if (rain.ndim != 3 or x.ndim != 3 or rain.shape[:2] != x.shape[:2]
                or 0 in rain.shape or 0 in x.shape):
            raise ValueError("rain (season,day,site) and predictors (season,day,covariate) must match")
        if not np.isfinite(x).all():
            raise ValueError("predictors must be finite; impute or remove missing seasons explicitly")
        if (np.isfinite(rain) & (rain < 0)).any() or not np.isfinite(rain).any():
            raise ValueError("rain must contain finite, nonnegative observations")
        if self.K > rain.shape[0] * rain.shape[1]:
            raise ValueError("n_states cannot exceed the number of observed days")
        if self.amount_predictor is not None and not -x.shape[-1] <= self._ai < x.shape[-1]:
            raise ValueError("amount_predictor index is outside the predictor array")
        return rain, x

    # --- pieces ---
    def _trans(self, x):
        """x (..., T, q) -> log transition (..., T, K, K)."""
        logits = self.A_[None, :, :] + (x @ self.lam_.T)[..., None, :]      # (..., T, K(from), K(to))
        return logits - logsumexp(logits, axis=-1, keepdims=True)

    def _log_emission(self, rain, x=None):
        """rain (Y, T, S), predictors (Y, T, q) -> (Y, T, K)."""
        ok = np.isfinite(rain)
        wet = ok & (rain >= self.thr)
        lp, lq = np.log(self.p_), np.log1p(-self.p_)                            # (K, S)
        e = np.einsum("yts,ks->ytk", wet.astype(float), lp) + np.einsum("yts,ks->ytk", (ok & ~wet).astype(float), lq)
        if self.amounts:
            x = np.where(wet, np.maximum(rain - self.thr, 0.01), 1.0)
            xa = x
            k = self.shape_                                                       # (K, S)
            lth = np.log(self.scale_)[None, None]                                 # (1,1,K,S)
            if self.amount_predictor is not None and self._x is not None:
                lth = lth + (self.g_[None, None, :] * self._x[..., self._ai][..., None])[..., None]
            logpdf = ((k[None, None] - 1) * np.log(xa)[..., None, :] - xa[..., None, :] * np.exp(-lth)
                      - gammaln(k)[None, None] - k[None, None] * lth)             # (Y,T,K,S)
            e = e + np.where(wet[..., None, :], logpdf, 0.0).sum(-1)
        return e

    def _forward_backward(self, le, lt):
        """le (Y,T,K), lt (Y,T,K,K) -> gamma (Y,T,K), xi (Y,T-1,K,K), loglik."""
        Y, T, K = le.shape
        la = np.zeros((Y, T, K))
        la[:, 0] = np.log(self.pi0_)[None] + le[:, 0]
        for t in range(1, T):
            la[:, t] = logsumexp(la[:, t - 1][:, :, None] + lt[:, t], axis=1) + le[:, t]
        lb = np.zeros((Y, T, K))
        for t in range(T - 2, -1, -1):
            lb[:, t] = logsumexp(lt[:, t + 1] + (le[:, t + 1] + lb[:, t + 1])[:, None, :], axis=2)
        ll = logsumexp(la[:, -1], axis=1)
        g = np.exp(la + lb - ll[:, None, None])
        xi = np.exp(la[:, :-1, :, None] + lt[:, 1:] + (le[:, 1:] + lb[:, 1:])[:, :, None, :] - ll[:, None, None, None])
        return g, xi, float(ll.sum())

    def _init_emissions(self, rain, x):
        """k-means start for the emission parameters (state 0 = driest)."""
        Y, T, S = rain.shape
        K = self.K
        rng = np.random.default_rng(self.seed)
        ok = np.isfinite(rain)
        wet = (ok & (rain >= self.thr)).astype(float)
        flat = np.where(ok, wet, np.nanmean(np.where(ok, wet, np.nan))).reshape(-1, S)
        cent = flat[rng.choice(len(flat), K, replace=False)]
        for _ in range(25):
            lab = np.argmin(((flat[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
            cent = np.stack([flat[lab == k].mean(0) if (lab == k).any() else cent[k] for k in range(K)])
        cent = cent[np.argsort(cent.mean(1))]
        self.p_ = np.clip(cent, 0.02, 0.98)
        positive = (rain - self.thr)[wet.astype(bool)]
        m_all = float(positive.mean()) if positive.size else 1.0
        self.shape_ = np.full((K, S), 0.8)
        self.scale_ = np.full((K, S), max(m_all, 1.0) / 0.8)
        self.g_ = np.zeros(K)
        self._x = x
        return rng

    def _m_step_emissions(self, g, rain, x):
        """Occurrence and (predictor-scaled) Gamma amounts from state posteriors g (Y, T, K)."""
        ok = np.isfinite(rain)
        pos = ok & (rain >= self.thr)
        wet = pos.astype(float)
        w_ok = np.einsum("ytk,yts->ks", g, ok.astype(float))
        self.p_ = np.clip(np.einsum("ytk,yts->ks", g, wet) / np.maximum(w_ok, 1e-9), 1e-3, 1 - 1e-3)
        if not self.amounts:
            return
        xw = np.where(pos, np.maximum(rain - self.thr, 0.01), 1.0)
        w = np.einsum("ytk,yts->ytks", g, pos.astype(float))
        sw = w.sum((0, 1))
        x1 = x[..., self._ai][..., None] if self.amount_predictor is not None else np.zeros(x.shape[:2] + (1,))
        for _ in range(3):                                           # alternate (k, theta) and g
            adj = xw[:, :, None, :] * np.exp(-self.g_[None, None, :, None] * x1[..., None])  # x e^{-g x_t}
            mean = (w * adj).sum((0, 1)) / np.maximum(sw, 1e-9)
            mlog = (w * np.log(adj)).sum((0, 1)) / np.maximum(sw, 1e-9)
            s_ = np.clip(np.log(np.maximum(mean, 1e-6)) - mlog, 1e-4, None)
            k = (3 - s_ + np.sqrt((s_ - 3) ** 2 + 24 * s_)) / (12 * s_)
            for _ in range(10):
                k = np.maximum(k - (np.log(k) - digamma(k) - s_) / (1 / k - polygamma(1, k)), 1e-3)
            self.shape_, self.scale_ = k, np.maximum(mean, 1e-3) / k
            if self.amount_predictor is None:
                break
            # Newton step for g_k: d/dg sum w [-k (log th + g x) - x e^{-g x}/th]
            xt = x1[..., None]                                           # (Y,T,1,1)
            e = xw[:, :, None, :] * np.exp(-self.g_[None, None, :, None] * xt) / self.scale_[None, None]
            grad = (w * (-self.shape_[None, None] * xt + e * xt)).sum((0, 1, 3))
            hess = -(w * e * xt ** 2).sum((0, 1, 3))
            self.g_ = np.clip(self.g_ - grad / np.minimum(hess, -1e-9), -3, 3)

    def fit(self, rain, predictors):
        """rain (Y, T, S) observed daily seasons; predictors (Y, T, q)."""
        rain, x = self._validated_inputs(rain, predictors)
        self.valid_sites_ = np.isfinite(rain).any((0, 1))
        q = x.shape[-1]
        K = self.K
        rng = self._init_emissions(rain, x)
        self.A_ = np.zeros((K, K)); self.A_[:, 1:] = rng.normal(0, 0.1, (K, K - 1))
        self.A_ += np.eye(K) * 1.5                                              # persistence prior start
        self.A_[:, 0] = 0.0
        self.lam_ = np.zeros((K, q))
        self.pi0_ = np.full(K, 1.0 / K)
        prev = -np.inf
        self.loglik_ = []
        for it in range(self.n_iter):
            lt = self._trans(x)
            self._x = x
            g, xi, ll = self._forward_backward(self._log_emission(rain), lt)
            self.loglik_.append(ll)
            self.pi0_ = np.clip(g[:, 0].mean(0), 1e-6, None); self.pi0_ /= self.pi0_.sum()
            self._m_step_emissions(g, rain, x)
            # transition parameters: maximise sum xi log P
            gprev = g[:, :-1]                                                   # (Y,T-1,K)
            xs = x[:, 1:]

            def nll(theta):
                A = np.zeros((K, K)); A[:, 1:] = theta[:K * (K - 1)].reshape(K, K - 1)
                lam = np.zeros((K, q)); lam[1:] = theta[K * (K - 1):].reshape(K - 1, q)
                logits = A[None, None] + (xs @ lam.T)[..., None, :]
                lp = logits - logsumexp(logits, axis=-1, keepdims=True)
                P = np.exp(lp)
                f = -(xi * lp).sum()
                resid = xi - gprev[..., None] * P                              # (Y,T-1,K,K)
                gA = -resid.sum((0, 1))[:, 1:]
                gl = -np.einsum("ytj,ytq->jq", resid.sum(2), xs)[1:]
                return f, np.r_[gA.ravel(), gl.ravel()]

            theta0 = np.r_[self.A_[:, 1:].ravel(), self.lam_[1:].ravel()]
            res = minimize(nll, theta0, jac=True, method="L-BFGS-B")
            self.A_[:, 1:] = res.x[:K * (K - 1)].reshape(K, K - 1)
            self.lam_[1:] = res.x[K * (K - 1):].reshape(K - 1, q)
            if abs(ll - prev) < self.tol * abs(ll):
                break
            prev = ll
        self.gamma_ = g
        # The M-step changes the parameters after the last recorded E-step.
        # Store posteriors for the returned fitted model, not the previous iterate.
        self._x = x
        self.gamma_, _, final_ll = self._forward_backward(self._log_emission(rain), self._trans(x))
        self.loglik_.append(final_ll)
        return self

    def viterbi(self, rain, predictors):
        """Most likely state sequence (Y, T)."""
        self._x = np.asarray(predictors, float)
        le = self._log_emission(np.asarray(rain, float))
        lt = self._trans(self._x)
        Y, T, K = le.shape
        d = np.log(self.pi0_)[None] + le[:, 0]
        back = np.zeros((Y, T, K), dtype=int)
        for t in range(1, T):
            cand = d[:, :, None] + lt[:, t]
            back[:, t] = np.argmax(cand, axis=1)
            d = cand.max(1) + le[:, t]
        path = np.zeros((Y, T), dtype=int)
        path[:, -1] = np.argmax(d, axis=1)
        for t in range(T - 2, -1, -1):
            path[:, t] = back[np.arange(Y), t + 1, path[:, t + 1]]
        return path

    def fit_spatial(self, rain, predictors, lat, lon, max_pairs=300, seed=0):
        """Within-state spatial dependence (optional): distance kernels fitted to
        occurrence and amounts on days of the decoded (Viterbi) states, with
        each state's own wet probability and scale removed as far as a kernel
        on raw data allows. ``simulate`` then draws spatially correlated
        uniforms instead of independent ones. Without it the NHMM
        underestimates short-range correlation (classical conditional
        independence)."""
        rain = np.asarray(rain, float)
        states = self.viterbi(rain, predictors)
        mixed = (self.p_ > 0.1) & (self.p_ < 0.9)                             # states with informative occurrence
        use = mixed.any(1)[states] if mixed.any() else np.ones(states.shape, bool)
        occ = np.where(np.isfinite(rain) & use[..., None], (rain >= self.thr).astype(float), np.nan).reshape(-1, rain.shape[-1])
        wet = np.isfinite(rain) & (rain >= self.thr)
        amt = np.where(wet, (rain - self.thr) / self.scale_[states], np.nan).reshape(-1, rain.shape[-1])
        self.spatial_models_ = {
            "occurrence": fit_distance_model(occ, lat, lon, kind="power", transform="binary", max_pairs=max_pairs, seed=seed),
            "amount": fit_distance_model(amt, lat, lon, kind="exponential", transform="gaussian", max_pairs=max_pairs, seed=seed)}
        self.lat_, self.lon_ = np.asarray(lat, float), np.asarray(lon, float)
        return self

    def _sample_states(self, x, rng):
        """First-order non-homogeneous state sequences (M, T)."""
        M, T, _ = x.shape
        P = np.exp(self._trans(x))                                             # (M,T,K,K)
        states = np.zeros((M, T), dtype=int)
        states[:, 0] = rng.choice(self.K, size=M, p=self.pi0_)
        for t in range(1, T):
            cp = np.cumsum(P[np.arange(M), t, states[:, t - 1]], axis=1)
            states[:, t] = (rng.random((M, 1)) > cp).sum(1).clip(max=self.K - 1)
        return states

    def simulate(self, predictors, n_sim=1, seed=None, trace=None):
        """predictors (M, T, q) -> rain (M * n_sim, T, S) and states.

        ``trace``: optional (p, mean) arrays (S,) for sub-threshold rain on dry days.
        Uses spatially correlated uniforms when ``fit_spatial`` was called."""
        x = np.asarray(predictors, float)
        if x.ndim == 2:
            x = x[None]
        if not hasattr(self, "p_"):
            raise RuntimeError("Call fit before simulate")
        if x.ndim != 3 or 0 in x.shape or x.shape[-1] != self.lam_.shape[-1] or not np.isfinite(x).all():
            raise ValueError("predictors must be finite (member,day,covariate) matching the fitted covariates")
        if not isinstance(n_sim, (int, np.integer)) or n_sim < 1:
            raise ValueError("n_sim must be a positive integer")
        x = np.repeat(x, n_sim, axis=0)
        M, T, _ = x.shape
        S = self.p_.shape[1]
        rng = np.random.default_rng(self.seed if seed is None else seed)
        states = self._sample_states(x, rng)
        p = self.p_[states]                                                    # (M,T,S)
        sp = getattr(self, "spatial_models_", None)
        if sp is not None:
            base = self.seed if seed is None else seed
            f_occ = GaussianSpatialField(self.lat_, self.lon_, model=sp["occurrence"], seed=base, n_features=256)
            f_amt = GaussianSpatialField(self.lat_, self.lon_, model=sp["amount"], seed=base + 1, n_features=256)
            u = np.stack([ndtr(f_occ.sample(M, t, 0)) for t in range(T)], 1)
            ua = np.clip(np.stack([ndtr(f_amt.sample(M, t, 1)) for t in range(T)], 1), 1e-9, 1 - 1e-9)
        else:
            u, ua = rng.random((M, T, S)), rng.random((M, T, S))
        wet = u < p
        # With amounts=False only occurrence was fitted; Gamma initialization
        # is not an estimated amount model. Represent wet days by the threshold.
        amt = self.thr
        if self.amounts:
            scale = self.scale_[states]
            if self.amount_predictor is not None:
                scale = scale * np.exp(self.g_[states] * x[..., self._ai])[..., None]
            amt = self.thr + gammaincinv(self.shape_[states], ua) * scale
        dry = 0.0
        if trace is not None:
            tp, tm = (np.asarray(v, float)[None, None] for v in trace)
            v = rng.random((M, T, S))
            dry = np.where(v < tp, np.minimum(2 * tm * v / np.maximum(tp, 1e-9), self.thr * (1 - 1e-6)), 0.0)
        rain = np.where(wet, amt, dry)
        rain[..., ~self.valid_sites_] = np.nan
        return rain, states

    def dwell_pmf(self, predictors=None, n_max=60):
        """Dwell-time pmf (K, n_max) of each state for constant predictors
        (default all zero): geometric, 1 - P_kk, for the first-order chain."""
        q = self.lam_.shape[1]
        x = np.zeros(q) if predictors is None else np.asarray(predictors, float)
        stay = np.diag(np.exp(self._trans(x[None])[0]))                      # (K,)
        n = np.arange(1, n_max + 1)
        return (1 - stay)[:, None] * stay[:, None] ** (n - 1)[None]


class NHSMM(NHMM):
    """Non-homogeneous hidden semi-Markov model (explicit-duration NHMM).

    The first-order hidden chain of ``NHMM`` gives geometric state durations,
    so long dry spells are too short. Here each state has its own dwell-time
    distribution through a discrete hazard (Guédon 2003; Langrock & Zucchini
    2011, hidden Markov representation of an HSMM with expanded states (k, d)):

        P(leave k after d days | still in k, x_t)
            = h_k(d, x_t) = sigmoid(sum_p c_kp (log d / log D)^p + beta_k . x_t),

    with d = 1..D (``max_duration``; ages >= D share the hazard at D, a
    geometric right tail). On leaving k, the next state j != k is chosen with
    softmax_{j != k}(A_kj + lambda_j . x_t), the NHMM transition restricted to
    j != k. The predictors x_t thus move both *how long* a regime lasts
    (beta) and *where it goes* (lambda). Emissions (multi-site occurrence,
    Gamma amounts scaled by exp(g_k x_ta)) are those of ``NHMM``. The day-0
    state age is drawn from the equilibrium age distribution of each state
    (proportional to the survival function) under the day-0 hazard.

    Fitted by EM on the expanded (K x D) chain (scaled forward-backward,
    vectorized over years). M-step: emissions as in ``NHMM``; hazards by
    weighted logistic Newton steps per state (expected leave/stay counts);
    destinations by weighted multinomial logit. ``init_iter`` NHMM iterations
    give the starting emissions and transitions. ``viterbi``, ``fit_spatial``,
    ``simulate`` and ``dwell_pmf`` work as for ``NHMM``.
    Sansom & Thomson (2001) used an HSMM for rainfall in the same spirit.

    The hazard update optimizes transition counts; it does not include the
    hazard-dependent equilibrium initial-age term. With this initialization
    the algorithm is an approximate EM procedure, and strict monotonicity of
    the observed-data likelihood is not guaranteed.
    """

    def __init__(self, n_states=4, max_duration=60, hazard_degree=2, hazard_predictors=True,
                 amounts=True, amount_predictor=-1, wet_threshold=1.0, n_iter=60, tol=1e-5,
                 init_iter=15, ridge=0.1, seed=0):
        super().__init__(n_states, amounts, amount_predictor, wet_threshold, n_iter, tol, seed)
        if self.K < 2:
            raise ValueError("NHSMM needs at least 2 states")
        self.D = int(max_duration)
        if self.D < 2:
            raise ValueError("max_duration must be >= 2")
        self.hazard_degree, self.hazard_predictors = int(hazard_degree), bool(hazard_predictors)
        self.init_iter, self.ridge = int(init_iter), float(ridge)
        if self.hazard_degree < 0 or self.init_iter < 0 or not np.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError("hazard_degree, init_iter and ridge must be nonnegative")

    # --- duration pieces ---
    def _basis(self):
        """(D, 1 + degree) polynomial basis in scaled log-duration."""
        ld = np.log(np.arange(1, self.D + 1.0)) / np.log(self.D)
        return np.stack([ld ** p for p in range(self.hazard_degree + 1)], -1)

    def _hazard(self, x):
        """x (..., T, q) -> leave probability (..., T, K, D)."""
        eta = (self.c_ @ self._basis().T)                                      # (K, D)
        eta = np.broadcast_to(eta, x.shape[:-1] + eta.shape)
        if self.hazard_predictors:
            eta = eta + (x @ self.beta_.T)[..., None]
        return np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-6, 1 - 1e-6)

    def _dest(self, x):
        """x (..., T, q) -> destination probabilities (..., T, K, K), zero diagonal."""
        logits = self.A_ + (x @ self.lam_.T)[..., None, :]
        logits = np.where(np.eye(self.K, dtype=bool), -np.inf, logits)
        return np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))

    def _initial(self, x0):
        """x0 (..., q) -> day-0 distribution over (K, D): pi_k x equilibrium age."""
        h = self._hazard(x0[..., None, :])[..., 0, :, :]                      # (..., K, D)
        age = np.concatenate([np.ones(h.shape[:-1] + (1,)), np.cumprod(1 - h[..., :-1], -1)], -1)
        age[..., -1] /= h[..., -1]
        age /= age.sum(-1, keepdims=True)
        return self.pi_[..., :, None] * age

    def _fb(self, le, H, Q, init):
        """Scaled forward-backward on the expanded chain.

        le (Y,T,K), H (Y,T,K,D), Q (Y,T,K,K), init (Y,K,D) ->
        alpha, beta, c, e, loglik (alpha * beta = posterior of (k, d))."""
        Y, T, K = le.shape
        D = self.D
        mx = le.max(-1, keepdims=True)
        e = np.exp(le - mx)
        a = np.empty((Y, T, K, D))
        c = np.empty((Y, T))
        a0 = init * e[:, 0, :, None]
        c[:, 0] = a0.sum((1, 2))
        a[:, 0] = a0 / c[:, 0, None, None]
        for t in range(1, T):
            prev, h = a[:, t - 1], H[:, t]
            stay = prev * (1 - h)
            pr = np.empty((Y, K, D))
            pr[:, :, 0] = np.einsum("yi,yij->yj", (prev * h).sum(-1), Q[:, t])
            pr[:, :, 1:] = stay[:, :, :-1]
            pr[:, :, -1] += stay[:, :, -1]
            at = pr * e[:, t, :, None]
            c[:, t] = at.sum((1, 2))
            a[:, t] = at / c[:, t, None, None]
        b = np.ones((Y, T, K, D))
        for t in range(T - 1, 0, -1):
            eb = e[:, t, :, None] * b[:, t]
            arr = np.einsum("yij,yj->yi", Q[:, t], eb[:, :, 0])
            nxt = np.concatenate([eb[:, :, 1:], eb[:, :, -1:]], -1)
            b[:, t - 1] = (H[:, t] * arr[:, :, None] + (1 - H[:, t]) * nxt) / c[:, t, None, None]
        ll = float((np.log(c).sum(1) + mx[..., 0].sum(1)).sum())
        return a, b, c, e, ll

    def _expected_counts(self, a, b, c, e, H, Q):
        """Leave / stay counts (Y,T-1,K,D) and destination counts (Y,T-1,K,K)."""
        eb = e[:, 1:, :, None] * b[:, 1:]                                        # (Y,T-1,K,D)
        cc = c[:, 1:, None, None]
        arr = np.einsum("ytij,ytj->yti", Q[:, 1:], eb[..., 0])
        nxt = np.concatenate([eb[..., 1:], eb[..., -1:]], -1)
        ah = a[:, :-1] * H[:, 1:]
        w_leave = ah * arr[..., None] / cc
        w_stay = a[:, :-1] * (1 - H[:, 1:]) * nxt / cc
        xi = ah.sum(-1)[..., :, None] * Q[:, 1:] * eb[:, :, None, :, 0] / cc
        return w_leave, w_stay, xi

    def _m_step_hazard(self, w_leave, w_stay, xs, max_newton=30):
        """Weighted logistic regression per state: leave vs stay, features
        (duration basis, predictors), Gaussian prior ``ridge`` on all but the
        intercept. Damped Newton iterations to convergence (the penalized
        objective is concave), so each M-step really maximizes."""
        B = self._basis()                                                       # (D, b)
        nb = B.shape[1]
        q = xs.shape[-1] if self.hazard_predictors else 0
        for k in range(self.K):
            wl, n = w_leave[:, :, k], w_leave[:, :, k] + w_stay[:, :, k]       # (Y,T-1,D)
            theta = np.r_[self.c_[k], self.beta_[k] if q else []]
            pen = np.full(theta.size, self.ridge); pen[0] = 1e-6

            def eta_of(th):
                e = (B @ th[:nb])[None, None]
                return e + (xs @ th[nb:])[..., None] if q else np.broadcast_to(e, wl.shape)

            def objective(th):
                e = eta_of(th)
                return float((wl * e - n * np.logaddexp(0.0, e)).sum() - 0.5 * (pen * th ** 2).sum())

            f = objective(theta)
            for _ in range(max_newton):
                p = 1.0 / (1.0 + np.exp(-eta_of(theta)))
                r, w = wl - n * p, n * p * (1 - p)
                wd = w.sum((0, 1))
                grad = [B.T @ r.sum((0, 1))]
                Hcc = (B * wd[:, None]).T @ B
                if q:
                    grad.append(np.einsum("yt,ytq->q", r.sum(-1), xs))
                    Hcb = B.T @ np.einsum("ytd,ytq->dq", w, xs)
                    Hbb = np.einsum("yt,ytq,ytr->qr", w.sum(-1), xs, xs)
                    Hm = np.block([[Hcc, Hcb], [Hcb.T, Hbb]])
                else:
                    Hm = Hcc
                grad = np.concatenate(grad) - pen * theta
                step = np.linalg.solve(Hm + np.diag(pen) + 1e-9 * np.eye(theta.size), grad)
                t = 1.0
                while t > 1e-4:
                    cand = theta + t * step
                    fc = objective(cand)
                    if fc >= f - 1e-10:
                        break
                    t *= 0.5
                else:
                    break
                theta, f_old, f = cand, f, fc
                if abs(f - f_old) < 1e-8 * max(abs(f), 1.0) and np.abs(t * step).max() < 1e-6:
                    break
            self.c_[k] = theta[:nb]
            if q:
                self.beta_[k] = theta[nb:]

    def _m_step_dest(self, xi, xs):
        """Weighted multinomial logit for the destination of a state change."""
        K, q = self.K, xs.shape[-1]
        if K == 2:
            return
        off = ~np.eye(K, dtype=bool)
        tot = xi.sum(-1)                                                        # (Y,T-1,K)

        def nll(theta):
            A = np.zeros((K, K)); A[off] = theta[:K * (K - 1)]
            lam = theta[K * (K - 1):].reshape(K, q)
            logits = np.where(~off, -np.inf, A + (xs @ lam.T)[..., None, :])
            lp = logits - logsumexp(logits, axis=-1, keepdims=True)
            P = np.exp(lp)
            f = -(xi * np.where(off, lp, 0.0)).sum() + 0.5 * self.ridge * (theta ** 2).sum()
            resid = xi - tot[..., None] * P
            gA = -resid.sum((0, 1))[off]
            gl = -np.einsum("ytj,ytq->jq", resid.sum(2), xs)
            return f, np.r_[gA, gl.ravel()] + self.ridge * theta

        theta0 = np.r_[self.A_[off], self.lam_.ravel()]
        res = minimize(nll, theta0, jac=True, method="L-BFGS-B")
        self.A_ = np.zeros((K, K)); self.A_[off] = res.x[:K * (K - 1)]
        self.lam_ = res.x[K * (K - 1):].reshape(K, q)

    def _pieces(self, x):
        return self._hazard(x), self._dest(x), self._initial(x[:, 0])

    # --- fitting ---
    def fit(self, rain, predictors):
        """rain (Y, T, S) observed daily seasons; predictors (Y, T, q)."""
        rain, x = self._validated_inputs(rain, predictors)
        self.valid_sites_ = np.isfinite(rain).any((0, 1))
        K, q, nb = self.K, x.shape[-1], self.hazard_degree + 1
        if self.init_iter > 0:
            base = NHMM(K, self.amounts, self.amount_predictor, self.thr, self.init_iter, self.tol, self.seed).fit(rain, x)
            for name in ("p_", "shape_", "scale_", "g_"):
                setattr(self, name, getattr(base, name).copy())
            self._x = x
            P = np.exp(base._trans(x))                                          # (Y,T,K,K)
            stay = np.einsum("ytk,ytkk->k", base.gamma_, P) / base.gamma_.sum((0, 1))
            self.A_, self.lam_ = base.A_.copy(), base.lam_.copy()
            self.pi_ = base.pi0_.copy()
            self.nhmm_init_ = base
        else:
            self._init_emissions(rain, x)
            stay = np.full(K, 0.7)
            self.A_, self.lam_ = np.zeros((K, K)), np.zeros((K, q))
            self.pi_ = np.full(K, 1.0 / K)
        self.c_ = np.zeros((K, nb))
        self.c_[:, 0] = np.log((1 - stay) / np.clip(stay, 1e-3, None))
        self.beta_ = np.zeros((K, q))
        xs = x[:, 1:]
        prev = -np.inf
        self.loglik_ = []
        for it in range(self.n_iter):
            self._x = x
            le = self._log_emission(rain)
            H, Q, init = self._pieces(x)
            a, b, c, e, ll = self._fb(le, H, Q, init)
            self.loglik_.append(ll)
            post = a * b                                                        # (Y,T,K,D)
            g = post.sum(-1)
            self.pi_ = np.clip(g[:, 0].mean(0), 1e-6, None); self.pi_ /= self.pi_.sum()
            self._m_step_emissions(g, rain, x)
            w_leave, w_stay, xi = self._expected_counts(a, b, c, e, H, Q)
            self._m_step_hazard(w_leave, w_stay, xs)
            self._m_step_dest(xi, xs)
            if abs(ll - prev) < self.tol * abs(ll):
                break
            prev = ll
        self._x = x
        a, b, _, _, final_ll = self._fb(self._log_emission(rain), *self._pieces(x))
        post = a * b
        self.gamma_ = post.sum(-1)
        self.loglik_.append(final_ll)
        self.duration_posterior_ = post.sum((0, 1))                              # expected days in (k, age)
        self.pi0_ = self.pi_
        return self

    def loglik(self, rain, predictors):
        x = np.asarray(predictors, float)
        self._x = x
        return self._fb(self._log_emission(np.asarray(rain, float)), *self._pieces(x))[-1]

    # --- decoding, simulation, diagnostics ---
    def viterbi(self, rain, predictors):
        """Most likely state sequence (Y, T) on the expanded (state, age) chain."""
        x = np.asarray(predictors, float)
        self._x = x
        le = self._log_emission(np.asarray(rain, float))
        H, Q, init = self._pieces(x)
        lH, l1H = np.log(H), np.log1p(-H)
        with np.errstate(divide="ignore"):
            lQ = np.log(Q)
        Y, T, K = le.shape
        D = self.D
        yy = np.arange(Y)[:, None]
        kk = np.arange(K)[None, :]
        delta = np.log(np.maximum(init, 1e-300)) + le[:, 0, :, None]
        back = np.zeros((Y, T, K, D), dtype=np.int32)
        for t in range(1, T):
            leave = delta + lH[:, t]
            dbest = leave.argmax(-1)                                            # (Y,K)
            cand = leave.max(-1)[:, :, None] + lQ[:, t]                         # (Y,K_from,K_to)
            ib = cand.argmax(1)                                                 # (Y,K_to)
            new = np.empty((Y, K, D))
            new[:, :, 0] = cand.max(1)
            back[:, t, :, 0] = ib * D + dbest[yy, ib]
            stay = delta + l1H[:, t]
            new[:, :, 1:-1] = stay[:, :, :-2]
            back[:, t, :, 1:-1] = kk[..., None] * D + np.arange(D - 2)[None, None]
            from_last = stay[:, :, -1] > stay[:, :, -2]
            new[:, :, -1] = np.where(from_last, stay[:, :, -1], stay[:, :, -2])
            back[:, t, :, -1] = kk * D + np.where(from_last, D - 1, D - 2)
            delta = new + le[:, t, :, None]
        flat = np.zeros((Y, T), dtype=np.int64)
        flat[:, -1] = delta.reshape(Y, -1).argmax(1)
        for t in range(T - 1, 0, -1):
            flat[:, t - 1] = back[:, t].reshape(Y, -1)[np.arange(Y), flat[:, t]]
        return flat // D

    def _sample_states(self, x, rng):
        """Semi-Markov state sequences (M, T): age-dependent leave, then destination."""
        M, T, _ = x.shape
        K, D = self.K, self.D
        H, Q, init = self._pieces(x)
        m = np.arange(M)
        s0 = (rng.random((M, 1)) > np.cumsum(init.reshape(M, -1), 1)).sum(1).clip(max=K * D - 1)
        k, d = s0 // D, s0 % D
        states = np.zeros((M, T), dtype=int)
        states[:, 0] = k
        for t in range(1, T):
            leave = rng.random(M) < H[m, t, k, d]
            j = (rng.random((M, 1)) > np.cumsum(Q[m, t, k], 1)).sum(1).clip(max=K - 1)
            k = np.where(leave, j, k)
            d = np.where(leave, 0, np.minimum(d + 1, D - 1))
            states[:, t] = k
        return states

    def dwell_pmf(self, predictors=None, n_max=None):
        """Dwell-time pmf (K, n_max) for constant predictors (default all zero)."""
        q = self.lam_.shape[1]
        x = np.zeros(q) if predictors is None else np.asarray(predictors, float)
        n_max = 2 * self.D if n_max is None else int(n_max)
        h = self._hazard(x[None])[0]                                            # (K, D)
        idx = np.minimum(np.arange(n_max), self.D - 1)
        hh = h[:, idx]
        surv = np.concatenate([np.ones((self.K, 1)), np.cumprod(1 - hh[:, :-1], 1)], 1)
        return surv * hh


def state_durations(states, n_states=None):
    """Run lengths of each hidden state in (n, T) state sequences -> list of arrays
    (runs cut by the season edges included)."""
    states = np.atleast_2d(states)
    K = int(states.max()) + 1 if n_states is None else int(n_states)
    out = [[] for _ in range(K)]
    for r in states:
        cut = np.flatnonzero(np.diff(r)) + 1
        for seg in np.split(r, cut):
            out[seg[0]].append(len(seg))
    return [np.asarray(v, int) for v in out]


def nhmm_predictors(ensemble_blocks, smooth=5, reference=None, harmonics=0, dates=None):
    """Daily predictors from a (year, member, day, site) model array.

    Returns (year, day, q): [standardized ensemble-mean domain-mean rainfall
    (running mean over ``smooth`` days), optional annual harmonics of the date
    (``harmonics``, needs ``dates``), standardized seasonal-mean anomaly (last)].
    ``reference`` = (mean, sd, season_mean, season_sd) from the hindcast so the
    forecast uses the same standardization; returned as second output.
    """
    x = np.nanmean(ensemble_blocks, axis=(1, 3))                               # (year, day)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        x = np.stack([np.convolve(np.pad(r, (smooth // 2, smooth - 1 - smooth // 2), mode="edge"), k, "valid") for r in x])
    seas = x.mean(1)
    if reference is None:
        reference = (x.mean(), x.std() + 1e-9, seas.mean(), seas.std() + 1e-9)
    m, s, sm, ss = reference
    cols = [(x - m) / s, np.broadcast_to(((seas - sm) / ss)[:, None], x.shape)]
    if harmonics:
        doy = (pd.DatetimeIndex(dates).dayofyear.to_numpy() if dates is not None else np.arange(x.shape[1])).astype(float)
        for k in range(1, harmonics + 1):
            cols += [np.broadcast_to(np.sin(2 * np.pi * k * doy / 365.25), x.shape),
                     np.broadcast_to(np.cos(2 * np.pi * k * doy / 365.25), x.shape)]
    pred = np.stack(cols[:1] + cols[2:] + cols[1:2], -1)          # seasonal anomaly kept LAST (amount predictor)
    return pred, reference


# ---------------------------------------------------------------------------
# 3. Dependence restoration
# ---------------------------------------------------------------------------
def stratified_quantiles(sample, M):
    """M representative values of a (N, ...) sample: means of M equal-probability bins.

    Equidistant quantiles i/(M+1) cut the upper tail of skewed daily rainfall
    and lose several percent of the mean; bin means keep the mean exactly.
    """
    M = int(M)
    x = np.asarray(sample, float)
    if x.ndim < 1 or x.shape[0] == 0 or M < 1:
        raise ValueError("sample must have a nonempty sample axis and M must be positive")
    x = np.sort(np.where(np.isfinite(x), x, np.nan), axis=0)
    N = x.shape[0]
    count = np.isfinite(x).sum(0)
    idx = np.arange(N).reshape((N,) + (1,) * (x.ndim - 1))
    out = []
    for i in range(M):
        lo, hi = count * i / M, count * (i + 1) / M
        w = np.clip(np.minimum(idx + 1, hi) - np.maximum(idx, lo), 0, None)
        denominator = w.sum(0)
        numerator = np.sum(w * np.nan_to_num(x), axis=0)
        out.append(np.divide(numerator, denominator, out=np.full(x.shape[1:], np.nan), where=denominator > 0))
    return np.stack(out)


def ensemble_copula_coupling(raw, calibrated, method="Q", rng=None):
    """ECC (Schefzik, Thorarinsdottir & Gneiting 2013).

    raw : (M, ...) raw ensemble supplying the rank (copula) structure
    calibrated : (N, ...) samples from the calibrated marginal of every column
        (N >= M; e.g. pooled corrected members of neighbouring days)
    method : 'Q' = M stratified quantiles (means of M equal-probability bins,
        mean-preserving); 'R' = M random draws.
    Ties in the raw ensemble (dry days) are broken at random.
    Returns (M, ...) with the calibrated marginals and the raw ranks.
    """
    rng = rng if rng is not None else np.random.default_rng()
    raw = np.asarray(raw, float)
    cal = np.asarray(calibrated, float)
    if raw.ndim < 1 or cal.ndim != raw.ndim or raw.shape[1:] != cal.shape[1:] or raw.shape[0] < 1 or cal.shape[0] < 1:
        raise ValueError("raw and calibrated need nonempty member axes and identical column dimensions")
    M = raw.shape[0]
    if method == "Q":
        values = stratified_quantiles(cal, M)
    elif method == "R":
        ordered = np.sort(np.where(np.isfinite(cal), cal, np.nan), axis=0)
        count = np.isfinite(ordered).sum(0)
        idx = (rng.random((M,) + cal.shape[1:]) * count[None]).astype(int)
        values = np.sort(np.take_along_axis(ordered, idx, axis=0), axis=0)
    else:
        raise ValueError("method must be 'Q' or 'R'")
    # An entirely missing raw column has no copula information. A partly
    # missing active column is rejected by schaake_shuffle rather than ranked
    # as artificial negative-infinite rainfall.
    values = np.where(np.isnan(raw).all(0)[None], np.nan, values)
    return schaake_shuffle(values, raw, axis=0, ties="random", rng=rng)


def preferential_dates(forecast_mean, obs_seasons, n, window=7, climatology=None):
    """Historical trajectories most similar to the forecast (after Scheuerer et al. 2017).

    forecast_mean : (T, S) forecast ensemble-mean (calibrated) field for each day
    obs_seasons   : (Y, T + 2*window, S) observed seasons padded by ``window`` days
    Each candidate trajectory (year y, shift o in [-window, window]) is scored by
    the mean squared difference of standardized anomalies (per site-day
    climatology of the candidates) between the forecast and the observed
    trajectory. The best shift of each year is taken first, so templates stay
    diverse; returns (year indices, offsets, scores) of the n best. Standard
    Schaake uses random candidates instead.
    """
    f = np.asarray(forecast_mean, float)
    obs = np.asarray(obs_seasons, float)
    Y, Tp, S = obs.shape
    T = f.shape[0]
    w, n = int(window), int(n)
    if f.ndim != 2 or f.shape[1] != S or w < 0 or Tp != T + 2 * w or Y < 1 or n < 1:
        raise ValueError("forecast (day,site), padded seasons (year,day+2*window,site) and positive n must agree")
    cands = [(y, o) for y in range(Y) for o in range(-w, w + 1)]
    traj = np.stack([obs[y, w + o:w + o + T] for y, o in cands])            # (C, T, S)
    active = np.isfinite(f)
    if not active.any():
        raise ValueError("forecast has no finite fields for selecting preferential templates")
    fields = traj[:, active]
    mu = np.nanmean(fields, 0) if climatology is None else np.asarray(climatology[0])[active]
    sd = (np.nanstd(fields, 0) if climatology is None else np.asarray(climatology[1])[active]) + 1e-6
    fa = (f[active] - mu) / sd
    oa = (fields - mu[None]) / sd[None]
    ok = np.isfinite(oa).all(1)
    score = np.where(ok, np.mean((oa - fa[None]) ** 2, axis=1), np.inf)
    order = np.argsort(score)
    # distinct years first (best shift of each year), then further shifts if n > years
    first, seen = [], set()
    for i in order:
        if cands[i][0] not in seen and np.isfinite(score[i]):
            first.append(i); seen.add(cands[i][0])
    selected = set(first)
    rest = [i for i in order if i not in selected and np.isfinite(score[i])]
    if not first:
        raise ValueError("No complete preferential template on the finite forecast fields")
    # Large ensembles may exceed the finite historical trajectory count.
    # Repeat the preferred candidates explicitly rather than returning fewer
    # rows than the marginal ensemble expects.
    best = np.resize(np.asarray(first + rest, dtype=int), n)
    return np.array([cands[i][0] for i in best]), np.array([cands[i][1] for i in best]), score[best]


# ---------------------------------------------------------------------------
# 4. Pipeline
# ---------------------------------------------------------------------------
class DynamicalDownscaler:
    """Bias-correct a daily dynamical ensemble and restore space-time structure.

    Steps: (1) per-member correction (``corrector``); (2) calibrated marginal
    at each cell/day = the corrected members of days t-``pool_days`` .. t+``pool_days``
    (a larger, smoother sample); (3) ``coupling``:
    ``"none"`` (keep corrected members as they are), ``"ecc"`` (ECC-Q on the
    raw member ranks), ``"schaake"`` (random observed template seasons),
    ``"preferential"`` (observed seasons most similar to the corrected forecast
    mean, Scheuerer et al. 2017).
    """

    def __init__(self, months=(7, 8, 9), corrector="loci_qm", coupling="ecc", pool_days=2, window=7,
                 wet_threshold=1.0, seed=42):
        if coupling not in {"none", "ecc", "schaake", "preferential"}:
            raise ValueError("coupling must be 'none', 'ecc', 'schaake' or 'preferential'")
        self.months, self.coupling, self.pool, self.window = validate_months(months), coupling, int(pool_days), int(window)
        self.thr, self.seed = float(wet_threshold), int(seed)
        if self.pool < 0 or self.window < 0 or not np.isfinite(self.thr) or self.thr <= 0:
            raise ValueError("pool_days/window must be nonnegative and wet_threshold positive")
        self.corrector = DailyBiasCorrector(corrector, wet_threshold) if isinstance(corrector, str) else corrector

    def _model_on_grid(self, model):
        _validate_grid(model, time=True)
        if not (np.array_equal(model.Y.values, self.y_) and np.array_equal(model.X.values, self.x_)):
            raise ValueError("Model Y,X coordinates must exactly match the fitted observation grid; regrid explicitly")
        rain = _convert_units(model, "PRCP")
        return rain.where(np.isfinite(rain) & (rain >= 0))

    def fit(self, hindcast: xr.DataArray, observations: xr.Dataset, years, template_years=None):
        """hindcast: daily PRCP (member, T, Y, X) on the observation grid (regrid
        first, e.g. ``hindcast.interp_like(obs, method='nearest')``); ``years`` =
        hindcast seasons (years of the first month) present in both, used to fit
        the corrector. ``template_years``: observed seasons allowed as Schaake
        templates (default: every complete observed season). ``downscale``
        never uses the target season itself as a template."""
        self.years_ = [int(y) for y in years]
        if not self.years_:
            raise ValueError("years must contain at least one complete calibration season")
        obs = canonicalize_observations(observations)["PRCP"]
        self.y_, self.x_ = obs["Y"].values, obs["X"].values
        mod = season_blocks(self._model_on_grid(hindcast), self.years_, self.months)  # (Y, M, T, S)
        ob = season_blocks(obs, self.years_, self.months)[:, 0]                 # (Y, T, S)
        self.month_ = season_dates(self.years_[0], self.months).month.to_numpy()
        self.corrector.fit(mod, ob, self.month_)
        # padded observed seasons for Schaake templates (all complete observed seasons)
        t_obs = pd.DatetimeIndex(obs["T"].values)
        candidates = range(t_obs.year.min(), t_obs.year.max() + 1) if template_years is None else [int(y) for y in template_years]
        all_years = [y for y in candidates if _padded_dates(y, self.months, self.window).isin(t_obs).all()]
        stacked = obs.assign_coords(T=t_obs.normalize()).stack(site=("Y", "X")).transpose("T", "site")
        pads = []
        for y in all_years:
            dates = _padded_dates(y, self.months, self.window)
            pads.append(stacked.sel(T=dates).values)
        self.obs_padded_ = (np.stack(pads) if pads else
                            np.empty((0, ob.shape[1] + 2 * self.window, ob.shape[2])))
        self.obs_years_ = all_years
        self.raw_hindcast_, self.obs_hindcast_ = mod, ob
        return self

    def _pooled(self, corrected):
        """(M, T, S) -> (M * (2p+1), T, S) pooled over +/- pool_days (edge days reuse edges)."""
        if self.pool == 0:
            return corrected
        T = corrected.shape[1]
        return np.concatenate([corrected[:, np.clip(np.arange(T) + k, 0, T - 1)] for k in range(-self.pool, self.pool + 1)], 0)

    def downscale(self, forecast: xr.DataArray, year, n_members=None):
        """forecast: daily PRCP (member, T, Y, X) on the observation grid for season ``year``."""
        if not hasattr(self, "years_"):
            raise RuntimeError("Call fit before downscale")
        raw = season_blocks(self._model_on_grid(forecast), [int(year)], self.months)[0]  # (M, T, S)
        M = raw.shape[0] if n_members is None else int(n_members)
        if M < 1:
            raise ValueError("n_members must be positive")
        dates = season_dates(int(year), self.months)
        corrected = self.corrector.transform(raw, dates.month.to_numpy())
        if np.shape(corrected) != raw.shape:
            raise ValueError("corrector.transform must return the same shape as its input")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(year), 17]))
        info = {"coupling": self.coupling}
        if self.coupling == "none":
            out = corrected[:M] if M <= corrected.shape[0] else corrected[rng.integers(0, corrected.shape[0], M)]
        else:
            pooled = self._pooled(corrected)
            if self.coupling == "ecc":
                if M > raw.shape[0]:
                    raise ValueError("ECC returns at most the raw ensemble size")
                out = ensemble_copula_coupling(raw[:M], pooled, "Q", rng)
            else:
                values = stratified_quantiles(pooled, M)
                keep = np.array([y != int(year) for y in self.obs_years_])      # never the target season itself
                pool_idx = np.flatnonzero(keep)
                if pool_idx.size == 0:
                    raise ValueError("No complete observed template seasons remain after excluding the forecast year")
                if self.coupling == "schaake":
                    C = pool_idx.size * (2 * self.window + 1)
                    pick = rng.choice(C, M, replace=M > C)
                    yi, oi = pool_idx[pick // (2 * self.window + 1)], pick % (2 * self.window + 1) - self.window
                else:
                    yi, oi, score = preferential_dates(np.nanmean(corrected, 0), self.obs_padded_[keep], M, self.window)
                    yi = pool_idx[yi]
                    info["template_score"] = score
                T = raw.shape[1]
                template = np.stack([self.obs_padded_[y, self.window + o:self.window + o + T] for y, o in zip(yi, oi)])
                ok = np.isfinite(template).all(0)
                values = np.where(ok[None], values, np.nan)
                template = np.where(ok[None], template, 0.0)
                out = schaake_shuffle(values, template, axis=0, ties="random", rng=rng)
                info["template_years"] = [self.obs_years_[y] for y in yi]
                info["template_offsets"] = oi
        dates = season_dates(int(year), self.months)
        ds = xr.Dataset({"PRCP": _to_dataarray(out.astype("float32"), dates, self.y_, self.x_, "PRCP")})
        ds.PRCP.attrs["units"] = "mm d-1"
        ds.attrs.update(generator="was-disaggregation dynamical downscaling", corrector=getattr(self.corrector, "method", type(self.corrector).__name__),
                        coupling=self.coupling, pool_days=self.pool)
        self.info_ = info
        return ds


# ---------------------------------------------------------------------------
# Synthetic stand-in for a C3S daily ensemble (demonstrations only)
# ---------------------------------------------------------------------------
def synthetic_model_ensemble(observations: xr.Dataset, years, months=(7, 8, 9), n_members=25, skill=0.6,
                             drizzle_prob=0.5, intensity_bias=0.75, smoothing=0.6, signal=None, year_error=0.12, seed=0):
    """ARTIFICIAL biased 'dynamical model' daily ensemble built from observations.

    For each season and member, the daily sequence of a random other year is
    taken, scaled towards the target season's observed total with strength
    ``skill``, spatially smoothed towards the domain mean, damped in intensity
    and given drizzle on dry days. These are the typical GCM biases (too many
    light-rain days, too little intensity, too smooth fields). ``signal``
    (year -> ratio) overrides the target ratio for years without observations
    (e.g. a forecast year). Returns DataArray (member, T, Y, X).
    """
    rng = np.random.default_rng(seed)
    if (n_members < 1 or not 0 <= skill <= 1 or not 0 <= drizzle_prob <= 1
            or not 0 <= smoothing <= 1 or intensity_bias <= 0 or year_error < 0):
        raise ValueError("invalid synthetic ensemble size, probability or bias parameters")
    pr = canonicalize_observations(observations)["PRCP"]
    t = pd.DatetimeIndex(pr["T"].values).normalize()
    pr = pr.assign_coords(T=t)
    stacked = pr.stack(site=("Y", "X")).transpose("T", "site")
    avail = [y for y in range(t.year.min(), t.year.max() + 1)
             if season_dates(y, months).isin(t).all()]
    seasons = {y: stacked.reindex(T=season_dates(y, months)).values for y in avail}
    def regional_total(v):
        complete = np.isfinite(v).all(0)
        return np.sum(v[:, complete], 0).mean() if complete.any() else np.nan
    clim = np.nanmean([regional_total(v) for v in seasons.values()]) if seasons else np.nan
    if not np.isfinite(clim) or clim <= 0:
        raise ValueError("synthetic_model_ensemble needs complete seasons with positive regional rainfall")
    arrays = []
    for y in years:
        d = season_dates(int(y), months)
        if y in seasons:
            ratio = regional_total(seasons[y]) / clim
        else:
            ratio = (signal or {}).get(int(y), 1.0)
        ratio = ratio * np.exp(rng.normal(0, year_error))            # imperfect seasonal signal
        mem = []
        donors = [a for a in avail if a != y]
        if not donors:
            raise ValueError(f"No complete donor season other than target year {y}")
        for m in range(n_members):
            donor = rng.choice(donors)
            x = seasons[donor][:len(d)].copy()
            r_d = regional_total(x) / clim
            x = x * (ratio / r_d) ** skill * np.exp(rng.normal(0, 0.1))
            x = smoothing * np.nanmean(x, axis=1, keepdims=True) + (1 - smoothing) * x
            x = x * intensity_bias
            dry = x < 1.0
            x = np.where(dry & (rng.random(x.shape) < drizzle_prob), rng.gamma(2.0, 0.8, x.shape), x)
            mem.append(x)
        arrays.append(xr.DataArray(np.stack(mem).reshape(n_members, len(d), pr.sizes["Y"], pr.sizes["X"]),
                                   dims=("member", "T", "Y", "X"),
                                   coords={"member": np.arange(n_members), "T": d, "Y": pr["Y"].values, "X": pr["X"].values}))
    out = xr.concat(arrays, dim="T").rename("PRCP")
    out.attrs.update(units="mm d-1", provenance="ARTIFICIAL model ensemble built from observations (was_disaggregation.dynamical.synthetic_model_ensemble)")
    return out


__all__ = ["DailyBiasCorrector", "ExternalCorrector", "NHMM", "NHSMM", "state_durations", "nhmm_predictors", "ensemble_copula_coupling",
           "preferential_dates", "DynamicalDownscaler", "synthetic_model_ensemble", "season_blocks"]
