"""Daily dynamical-ensemble downscaling and dependence restoration.

This module implements identifiable mathematical components from the references
below, together with explicitly named package variants. It does not reproduce
the articles' original datasets, fitted parameters or evaluation experiments.

* ``DailyBiasCorrector`` implements empirical quantile mapping (QM) and the
  wet-frequency/intensity LOCI mechanism of Schmidli, Frei and Vidale (2006).
  ``loci_qm`` combines the two mechanisms; it is a package hybrid.
* ``NHMM`` uses the predictor-dependent hidden-state framework of Hughes and
  Guttorp (1994), with the occurrence-only transition/emission structure of
  Robertson, Kirshner and Smyth (2004). Gamma amounts, spatial copulas and
  predictor-dependent initialization are package extensions.
* ``NHSMM`` uses an expanded state/age representation related to Langrock and
  Zucchini (2011). Guédon (2003) is a foundational semi-Markov reference, not
  the exact fitted algorithm here. The polynomial predictor-dependent hazard,
  geometric tail and conditional initial-age distribution are package choices.
* ``ensemble_copula_coupling`` implements ECC-Q and ECC-R from Schefzik,
  Thorarinsdottir and Gneiting (2013). ``Q-midpoint`` and ``Q-mean`` explicitly
  distinguish alternative marginal representatives from original ECC-Q.
* Historical reordering uses Clark, Gangopadhyay, Hay, Rajagopalan and Wilby
  (2004). ``minimum_divergence_selection`` implements the integrated-CDF
  divergence and one-at-a-time backward elimination of Scheuerer, Hamill,
  Whitin, He and Henkel (2017), using empirical predictive distributions.
  ``preferential_dates`` is a separate forecast-mean analogue heuristic.
* ``DynamicalDownscaler`` composes these components. ``synthetic_model_ensemble``
  creates artificial inputs for demonstrations; it is not an operational model.

Implemented safeguards include explicit ECC quantile conventions, interval
censoring near the rainfall threshold, generalized-EM objective ascent checks,
and an initial-age term in semi-Markov fitting. These improve mathematical or
numerical consistency. They do not establish superior out-of-sample forecast
skill. Remaining limits include tied-value wet-frequency errors, historical
support and stationarity assumptions, greedy MDSS selection and model-dependent
residual spatial calibration. See ``docs/MATH_DYNAMICAL_FR.md``.

References
----------
Hughes, James P.; Guttorp, Peter (1994). A class of stochastic models for
    relating synoptic atmospheric patterns to regional hydrologic phenomena.
    Water Resources Research, 30, 1535-1546. https://doi.org/10.1029/93WR02983
Robertson, Andrew W.; Kirshner, Sergey; Smyth, Padhraic (2004). Downscaling of
    daily rainfall occurrence over Northeast Brazil using a hidden Markov model.
    Journal of Climate, 17, 4407-4424. https://doi.org/10.1175/JCLI-3216.1
Guédon, Yann (2003). Estimating hidden semi-Markov chains from discrete sequences.
    Journal of Computational and Graphical Statistics, 12, 604-639.
    https://doi.org/10.1198/1061860032030
Langrock, Roland; Zucchini, Walter (2011). Hidden Markov models with arbitrary
    state dwell-time distributions. Computational Statistics & Data Analysis,
    55, 715-724. https://doi.org/10.1016/j.csda.2010.06.015
Schmidli, Jürg; Frei, Christoph; Vidale, Pier Luigi (2006). Downscaling from GCM
    precipitation: a benchmark for dynamical and statistical downscaling methods.
    International Journal of Climatology, 26, 679-689.
    https://doi.org/10.1002/joc.1287
Gudmundsson, Lukas; Bremnes, John Bjørnar; Haugen, Jan Erik; Engen-Skaugen,
    Torill (2012). Technical Note: Downscaling RCM precipitation to the station
    scale using statistical transformations - a comparison of methods.
    Hydrology and Earth System Sciences, 16, 3383-3390.
    https://doi.org/10.5194/hess-16-3383-2012
Clark, Martyn; Gangopadhyay, Subhrendu; Hay, Lauren; Rajagopalan, Balaji; Wilby,
    Robert (2004). The Schaake shuffle: a method for reconstructing space-time
    variability in forecasted precipitation and temperature fields.
    Journal of Hydrometeorology, 5, 243-262.
    https://doi.org/10.1175/1525-7541(2004)005<0243:TSSAMF>2.0.CO;2
Schefzik, Roman; Thorarinsdottir, Thordis L.; Gneiting, Tilmann (2013).
    Uncertainty quantification in complex simulation models using ensemble
    copula coupling. Statistical Science, 28, 616-640.
    https://doi.org/10.1214/13-STS443
Scheuerer, Michael; Hamill, Thomas M.; Whitin, Brett; He, Minxue; Henkel,
    Arthur (2017). A method for preferential selection of dates in the Schaake
    shuffle approach to constructing spatiotemporal forecast fields of
    temperature and precipitation. Water Resources Research, 53, 3029-3046.
    https://doi.org/10.1002/2016WR020133
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammainc, gammaincinv, gammaln, logsumexp

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
    ``method="loci"``: the model threshold t_m is set so that the model
    wet-day frequency approximates the observed frequency; rescale retained
    model wet days by their mean to match observed mean wet-day intensity. This is
    the local intensity scaling of Jürg Schmidli, Christoph Frei and Pier Luigi
    Vidale (2006), https://doi.org/10.1002/joc.1287. Only their simplest
    frequency/intensity mechanism is implemented; flow-dependent correction
    and the Alpine experiment are not reproduced.
    ``method="loci_qm"``: the model threshold t_m is set so that the model
    wet-day frequency (x > t_m) approximates the observed frequency (x >= wet_threshold);
    model days <= t_m become dry (0), and wet-day amounts are quantile-mapped
    onto observed wet-day amounts. If one or both wet-day calibration samples
    have fewer than five observations, a positive intensity scale is used in
    place of wet-day quantiles. Exact frequencies cannot generally be obtained
    with a deterministic threshold when model values are tied.

    QM follows the empirical-distribution transformation reviewed by Lukas
    Gudmundsson, John Bjørnar Bremnes, Jan Erik Haugen and Torill Engen-Skaugen
    (2012), https://doi.org/10.5194/hess-16-3383-2012. Quantile interpolation,
    upper-tail offset extrapolation, sparse-sample fallback and the hybrid
    ``loci_qm`` are implementation choices, not a unique algorithm from that
    review. No preservation of an unobserved climate-change signal is assumed.
    """

    def __init__(self, method="loci_qm", wet_threshold=1.0, n_quantiles=200):
        if method not in {"qm", "loci", "loci_qm"}:
            raise ValueError("method must be 'qm', 'loci' or 'loci_qm'")
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
        self.wet_scale_, self.loci_scale_, self.wet_fallback_ = {}, {}, {}
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
                    # The hybrid QM fallback rescales excesses when there are
                    # too few wet days to estimate a useful wet CDF.
                    mean_excess = np.mean(mw - t_mod[s]) if mw.size else np.nan
                    self.wet_fallback_.setdefault(m, np.full(ns, np.nan))[s] = (
                        ow.mean() if ow.size else np.nan)
                    self.wet_scale_.setdefault(m, np.full(ns, np.nan))[s] = (
                        ow.mean() / mean_excess if ow.size and np.isfinite(mean_excess)
                        and mean_excess > 1e-12 else np.nan)
                    # LOCI scales the original retained precipitation, not
                    # precipitation shifted by the occurrence cutoff.
                    self.loci_scale_.setdefault(m, np.full(ns, np.nan))[s] = (
                        ow.mean() / mw.mean() if ow.size and mw.size and mw.mean() > 1e-12
                        else np.nan)
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
                if self.method == "loci":
                    y = x * self.loci_scale_[m]
                    y = np.where(np.isfinite(self.loci_scale_[m]), y, self.wet_fallback_[m])
                    y = np.maximum(y, 0.0)
                else:
                    y = self._map(x, self.mod_q_[m], self.obs_q_[m])
                    sparse = ~np.isfinite(self.mod_q_[m]).all(1)
                    fallback = (x - self.mod_thr_[m]) * self.wet_scale_[m]
                    fallback = np.where(np.isfinite(self.wet_scale_[m]), fallback, self.wet_fallback_[m])
                    y = np.where(sparse, fallback, y)
                    y = np.maximum(y, self.thr)
                out[..., sel, :] = np.where(np.isfinite(x), np.where(wet, y, 0.0), np.nan)
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
    """Multi-site rainfall NHMM after James P. Hughes and Peter Guttorp (1994)
    and Andrew W. Robertson, Sergey Kirshner and Padhraic Smyth (2004).

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

    ``initial_predictors=True`` fits a regularized multinomial logistic model
    for the first-day state using the first-day predictors. The default uses
    unconditional initial state probabilities, as in previous releases.
    ``amount_resolution`` defines a left-censored interval for wet-day excess
    observations near zero; simulated amounts are the continuous latent values.
    Occurrence-only modeling (``amounts=False``) is the part corresponding to
    Robertson, Kirshner and Smyth (2004), https://doi.org/10.1175/JCLI-3216.1.
    Gamma amounts, Gaussian spatial kernels, interval censoring and regularized
    predictor-dependent initialization are package extensions. The emissions
    do not reproduce the autologistic spatial model of Hughes and Guttorp's
    separate 1994 spatial-dependence article. Numerical fitting safeguards
    check penalized objective ascent; they cannot rule out local optima or
    state-label ambiguity. See module references for the original NHMM source,
    https://doi.org/10.1029/93WR02983.
    """

    def __init__(self, n_states=4, amounts=True, amount_predictor=-1, wet_threshold=1.0, n_iter=60, tol=1e-4, seed=0,
                 initial_predictors=False, initial_ridge=1.0, amount_resolution=0.01):
        self.K, self.amounts, self.thr = int(n_states), bool(amounts), float(wet_threshold)
        # index of the predictor scaling wet-day amounts (None: no amount predictor)
        self.amount_predictor = amount_predictor
        self._ai = 0 if amount_predictor is None else int(amount_predictor)
        self.n_iter, self.tol, self.seed = int(n_iter), float(tol), int(seed)
        self.initial_predictors, self.initial_ridge = bool(initial_predictors), float(initial_ridge)
        self.amount_resolution = float(amount_resolution)
        if self.K < 2 or self.n_iter < 1 or self.tol < 0 or not np.isfinite(self.thr) or self.thr <= 0:
            raise ValueError("n_states must be >= 2, n_iter positive, tol nonnegative and wet_threshold positive")
        if not np.isfinite(self.initial_ridge) or self.initial_ridge < 0:
            raise ValueError("initial_ridge must be nonnegative and finite")
        if not np.isfinite(self.amount_resolution) or self.amount_resolution <= 0:
            raise ValueError("amount_resolution must be positive and finite")

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

    def _initial_probs(self, x0):
        """State probabilities for the first day, (season, K)."""
        x0 = np.asarray(x0, float)
        if not self.initial_predictors:
            return np.broadcast_to(self.pi0_, (len(x0), self.K))
        design = np.column_stack([np.ones(len(x0)), x0])
        logits = np.concatenate([np.zeros((len(x0), 1)), design @ self.initial_coef_], axis=1)
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))

    def _fit_initial(self, posterior, x0):
        """Regularized weighted logit on posterior first-day states."""
        if not self.initial_predictors:
            return
        design = np.column_stack([np.ones(len(x0)), x0])
        k = self.K
        ridge = np.ones((design.shape[1], k - 1)) * self.initial_ridge
        ridge[0] = 0.0

        def nll(flat):
            coef = flat.reshape(design.shape[1], k - 1)
            logits = np.concatenate([np.zeros((len(x0), 1)), design @ coef], axis=1)
            lp = logits - logsumexp(logits, axis=1, keepdims=True)
            p = np.exp(lp)
            loss = -(posterior * lp).sum() + 0.5 * (ridge * coef ** 2).sum()
            grad = design.T @ (p[:, 1:] - posterior[:, 1:]) + ridge * coef
            return loss, grad.ravel()

        result = minimize(nll, self.initial_coef_.ravel(), jac=True, method="L-BFGS-B")
        if not np.isfinite(result.fun):
            raise RuntimeError("Conditional initial state fit failed")
        self.initial_coef_ = result.x.reshape(design.shape[1], k - 1)

    def _log_emission(self, rain, x=None):
        """rain (Y, T, S), predictors (Y, T, q) -> (Y, T, K)."""
        predictors = self._x if x is None else np.asarray(x, float)
        ok = np.isfinite(rain)
        wet = ok & (rain >= self.thr)
        lp, lq = np.log(self.p_), np.log1p(-self.p_)                            # (K, S)
        e = np.einsum("yts,ks->ytk", wet.astype(float), lp) + np.einsum("yts,ks->ytk", (ok & ~wet).astype(float), lq)
        if self.amounts:
            excess = rain - self.thr
            censored = wet & (excess <= self.amount_resolution)
            x = np.where(wet, np.maximum(excess, self.amount_resolution), 1.0)
            xa = x
            k = self.shape_                                                       # (K, S)
            lth = np.log(self.scale_)[None, None]                                 # (1,1,K,S)
            if self.amount_predictor is not None and predictors is not None:
                lth = lth + (self.g_[None, None, :] * predictors[..., self._ai][..., None])[..., None]
            logpdf = ((k[None, None] - 1) * np.log(xa)[..., None, :] - xa[..., None, :] * np.exp(-lth)
                      - gammaln(k)[None, None] - k[None, None] * lth)             # (Y,T,K,S)
            # Threshold observations encode the interval [0, resolution] in
            # latent wet-day excess; replacing zero by a tiny point density
            # creates artificial likelihood singularities when shape < 1.
            limit = self.amount_resolution * np.exp(-lth)
            logcdf = np.log(np.maximum(gammainc(k[None, None], limit), 1e-300))
            logpdf = np.where(censored[..., None, :], logcdf, logpdf)
            e = e + np.where(wet[..., None, :], logpdf, 0.0).sum(-1)
        return e

    def _forward_backward(self, le, lt):
        """le (Y,T,K), lt (Y,T,K,K) -> gamma (Y,T,K), xi (Y,T-1,K,K), loglik."""
        Y, T, K = le.shape
        la = np.zeros((Y, T, K))
        start = self._initial_probs(self._x[:, 0]) if self.initial_predictors else self.pi0_[None]
        la[:, 0] = np.log(start) + le[:, 0]
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
        """Weighted Bernoulli MLE and bounded Gamma generalized M-step.

        Optimize the complete weighted amount log-likelihood jointly over
        each state's site parameters and its common predictor coefficient.
        Sites with no expected wet observations keep their previous parameters.
        """
        ok = np.isfinite(rain)
        pos = ok & (rain >= self.thr)
        w_ok = np.einsum("ytk,yts->ks", g, ok.astype(float))
        self.p_ = np.clip(np.einsum("ytk,yts->ks", g, pos.astype(float)) /
                          np.maximum(w_ok, 1e-12), 1e-3, 1 - 1e-3)
        if not self.amounts:
            return
        censored = pos & (rain - self.thr <= self.amount_resolution)
        z = np.where(pos, np.maximum(rain - self.thr, self.amount_resolution), 1.0)
        predictor = x[..., self._ai] if self.amount_predictor is not None else np.zeros(x.shape[:2])
        for state in range(self.K):
            weights = g[..., state, None] * pos
            active = weights.sum((0, 1)) > 1e-10
            if not active.any():
                continue
            w, amount = weights[..., active], z[..., active]
            interval = censored[..., active]
            log_amount = np.log(amount)
            ns = int(active.sum())
            coef = self.g_[state] if self.amount_predictor is not None else 0.0
            theta0 = np.r_[np.log(self.shape_[state, active]),
                           np.log(self.scale_[state, active]), coef]

            def objective(theta):
                shape, scale = np.exp(theta[:ns]), np.exp(theta[ns:2 * ns])
                log_scale = np.log(scale)[None, None] + theta[-1] * predictor[..., None]
                ratio = amount * np.exp(-log_scale)
                lp = ((shape - 1) * log_amount - ratio - gammaln(shape) - shape * log_scale)
                shape_score = shape * (log_amount - log_scale - digamma(shape))
                scale_score = ratio - shape
                if interval.any():
                    upper = self.amount_resolution * np.exp(-log_scale)
                    logcdf = np.log(np.maximum(gammainc(shape, upper), 1e-300))
                    relative_step = 1e-5
                    cdf_plus = np.log(np.maximum(gammainc(shape * (1 + relative_step), upper), 1e-300))
                    cdf_minus = np.log(np.maximum(gammainc(shape * (1 - relative_step), upper), 1e-300))
                    shape_score = np.where(interval, (cdf_plus - cdf_minus) / (2 * relative_step), shape_score)
                    density_ratio = np.exp(shape * np.log(upper) - upper - gammaln(shape) - logcdf)
                    scale_score = np.where(interval, -density_ratio, scale_score)
                    lp = np.where(interval, logcdf, lp)
                shape_gradient = (w * shape_score).sum((0, 1))
                scale_gradient = (w * scale_score).sum((0, 1))
                coef_gradient = (w * scale_score * predictor[..., None]).sum()
                return -float((w * lp).sum()), -np.r_[shape_gradient, scale_gradient, coef_gradient]

            bounds = [(np.log(1e-3), np.log(1e3))] * ns + [(np.log(1e-6), np.log(1e6))] * ns
            bounds += [(-3.0, 3.0) if self.amount_predictor is not None else (0.0, 0.0)]
            result = minimize(objective, theta0, method="L-BFGS-B", jac=True, bounds=bounds)
            if np.isfinite(result.fun) and result.fun <= objective(theta0)[0] + 1e-8:
                self.shape_[state, active] = np.exp(result.x[:ns])
                self.scale_[state, active] = np.exp(result.x[ns:2 * ns])
                self.g_[state] = result.x[-1]

    def _parameter_names(self):
        return ("A_", "lam_", "pi0_", "initial_coef_", "p_", "shape_", "scale_", "g_")

    def _penalty(self):
        if self.initial_predictors:
            return 0.5 * self.initial_ridge * float((self.initial_coef_[1:] ** 2).sum())
        return 0.0

    def _observed_loglik(self, rain, x):
        self._x = x
        return self._forward_backward(self._log_emission(rain, x), self._trans(x))[-1]

    def _accept_m_step(self, previous, rain, x, old_objective):
        """Safeguard generalized EM against numerical optimizer decreases.

        Penalized likelihood, not raw likelihood, is the ascent criterion when
        an explicit ridge prior is enabled. Convex parameter interpolation
        preserves probability normalization and the parameter bounds.
        """
        candidate = {name: getattr(self, name).copy() for name in previous}
        factor = 1.0
        for _ in range(24):
            score = self._observed_loglik(rain, x) - self._penalty()
            if np.isfinite(score) and score >= old_objective - 1e-8:
                return score
            factor *= 0.5
            for name, old in previous.items():
                setattr(self, name, old + factor * (candidate[name] - old))
        for name, old in previous.items():
            setattr(self, name, old)
        return old_objective

    def fit(self, rain, predictors):
        """rain (Y, T, S) observed daily seasons; predictors (Y, T, q)."""
        rain, x = self._validated_inputs(rain, predictors)
        self.valid_sites_ = np.isfinite(rain).any((0, 1))
        q = x.shape[-1]
        K = self.K
        rng = self._init_emissions(rain, x)
        self.A_ = np.zeros((K, K)); self.A_[:, 1:] = rng.normal(0, 0.1, (K, K - 1))
        self.A_ += np.eye(K) * 1.5                                              # persistent initial guess, not a prior
        self.A_[:, 0] = 0.0
        self.lam_ = np.zeros((K, q))
        self.pi0_ = np.full(K, 1.0 / K)
        self.initial_coef_ = np.zeros((q + 1, K - 1))
        prev = -np.inf
        self.loglik_ = []
        self.objective_ = []
        for it in range(self.n_iter):
            lt = self._trans(x)
            self._x = x
            g, xi, ll = self._forward_backward(self._log_emission(rain), lt)
            self.loglik_.append(ll)
            old_objective = ll - self._penalty()
            self.objective_.append(old_objective)
            previous = {name: getattr(self, name).copy() for name in self._parameter_names()}
            self.pi0_ = np.clip(g[:, 0].mean(0), 1e-6, None); self.pi0_ /= self.pi0_.sum()
            self._fit_initial(g[:, 0], x[:, 0])
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
            self._accept_m_step(previous, rain, x, old_objective)
            if abs(ll - prev) < self.tol * abs(ll):
                break
            prev = ll
        self.gamma_ = g
        # The M-step changes the parameters after the last recorded E-step.
        # Store posteriors for the returned fitted model, not the previous iterate.
        self._x = x
        self.gamma_, _, final_ll = self._forward_backward(self._log_emission(rain), self._trans(x))
        self.loglik_.append(final_ll)
        self.objective_.append(final_ll - self._penalty())
        return self

    def loglik(self, rain, predictors):
        """Observed-data log-likelihood of complete seasons, marginalizing states.

        Includes occurrence and fitted amounts; excludes explicit ridge priors.
        All finite observation values are conditioned on the supplied predictors.
        """
        if not hasattr(self, "p_"):
            raise RuntimeError("Call fit before loglik")
        rain, x = self._validated_inputs(rain, predictors)
        if rain.shape[-1] != self.p_.shape[1] or x.shape[-1] != self.lam_.shape[-1]:
            raise ValueError("rain sites and predictor count must match the fitted model")
        return self._observed_loglik(rain, x)

    def viterbi(self, rain, predictors):
        """Most likely state sequence (Y, T)."""
        self._x = np.asarray(predictors, float)
        le = self._log_emission(np.asarray(rain, float))
        lt = self._trans(self._x)
        Y, T, K = le.shape
        d = np.log(self._initial_probs(self._x[:, 0])) + le[:, 0]
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
        amount_scale = self.scale_[states]
        if self.amount_predictor is not None:
            x = np.asarray(predictors, float)
            amount_scale = amount_scale * np.exp(self.g_[states] * x[..., self._ai])[..., None]
        amt = np.where(wet, (rain - self.thr) / amount_scale, np.nan).reshape(-1, rain.shape[-1])
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
        p0 = self._initial_probs(x[:, 0])
        states[:, 0] = (rng.random((M, 1)) > np.cumsum(p0, axis=1)).sum(1).clip(max=self.K - 1)
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

    The homogeneous first-order hidden chain gives geometric state durations.
    This restriction can misrepresent persistence. Here each state has a dwell-time
    distribution through a discrete hazard. The general semi-Markov framework
    is described by Yann Guédon (2003); the expanded states (k, d) follow the
    representation principle of Roland Langrock and Walter Zucchini (2011):

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
    weighted logistic L-BFGS-B steps per state (expected leave/stay counts
    and the equilibrium initial-age term);
    destinations by weighted multinomial logit. ``init_iter`` NHMM iterations
    give the starting emissions and transitions. ``viterbi``, ``fit_spatial``,
    ``simulate`` and ``dwell_pmf`` work as for ``NHMM``.

    The hazard M-step includes both transition counts and the equilibrium
    initial-age contribution. Numerical generalized EM steps are safeguarded
    by ascent of the penalized observed-data likelihood (``objective_``).
    Conditional equilibrium at the first-day predictor is an explicit model
    assumption; it does not imply a stationary non-homogeneous process.

    References: Guédon (2003), https://doi.org/10.1198/1061860032030;
    Langrock and Zucchini (2011), https://doi.org/10.1016/j.csda.2010.06.015.
    The polynomial hazard, geometric right tail and fitting safeguards are
    package extensions. This is not Guédon's original discrete-sequence
    estimator or a reproduction of Langrock and Zucchini's rainfall experiment.
    """

    def __init__(self, n_states=4, max_duration=60, hazard_degree=2, hazard_predictors=True,
                 amounts=True, amount_predictor=-1, wet_threshold=1.0, n_iter=60, tol=1e-5,
                 init_iter=15, ridge=0.1, seed=0, initial_predictors=False, initial_ridge=1.0,
                 amount_resolution=0.01):
        super().__init__(n_states, amounts, amount_predictor, wet_threshold, n_iter, tol, seed,
                         initial_predictors, initial_ridge, amount_resolution)
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

    def _hazard_logits(self, x):
        eta = self.c_ @ self._basis().T
        eta = np.broadcast_to(eta, x.shape[:-1] + eta.shape)
        if self.hazard_predictors:
            eta = eta + (x @ self.beta_.T)[..., None]
        return eta

    def _hazard(self, x):
        """x (..., T, q) -> leave probability (..., T, K, D)."""
        return np.clip(expit(self._hazard_logits(x)), 1e-15, 1 - 1e-15)

    def _dest(self, x):
        """x (..., T, q) -> destination probabilities (..., T, K, K), zero diagonal."""
        logits = self.A_ + (x @ self.lam_.T)[..., None, :]
        logits = np.where(np.eye(self.K, dtype=bool), -np.inf, logits)
        return np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))

    @staticmethod
    def _age_logprobs(eta):
        """Equilibrium age probabilities, with the last age aggregating the tail."""
        log_stay = -np.logaddexp(0.0, eta)
        log_leave = -np.logaddexp(0.0, -eta)
        log_survival = np.concatenate([np.zeros(eta.shape[:-1] + (1,)),
                                       np.cumsum(log_stay[..., :-1], axis=-1)], axis=-1)
        log_survival[..., -1] -= log_leave[..., -1]
        return log_survival - logsumexp(log_survival, axis=-1, keepdims=True)

    def _initial(self, x0):
        """x0 (..., q) -> day-0 distribution over (K, D): pi_k x equilibrium age."""
        eta = self._hazard_logits(x0[..., None, :])[..., 0, :, :]
        age = np.exp(self._age_logprobs(eta))
        return self._initial_probs(x0)[..., :, None] * age

    def _initial_probs(self, x0):
        if self.initial_predictors:
            return super()._initial_probs(x0)
        return np.broadcast_to(self.pi_, (len(x0), self.K))

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

    def _hazard_objective(self, theta, state, w_leave, w_stay, xs, initial_posterior, x0):
        """Negative complete log-likelihood + ridge; returns analytic gradient.

        The initial-age term depends on the hazard. Omitting it gives an
        invalid EM M-step whenever initial ages use equilibrium survival.
        """
        basis = self._basis()
        nb = basis.shape[1]
        q = xs.shape[-1] if self.hazard_predictors else 0
        eta = np.broadcast_to((basis @ theta[:nb])[None, None], w_leave.shape)
        if q:
            eta = eta + (xs @ theta[nb:])[..., None]
        n = w_leave + w_stay
        residual = w_leave - n * expit(eta)
        loglik = float((w_leave * eta - n * np.logaddexp(0.0, eta)).sum())
        gradient = [basis.T @ residual.sum((0, 1))]
        if q:
            gradient.append(np.einsum("yt,ytq->q", residual.sum(-1), xs))
        gradient = np.concatenate(gradient)
        if initial_posterior is not None:
            eta0 = np.broadcast_to((basis @ theta[:nb])[None], initial_posterior.shape)
            if q:
                eta0 = eta0 + (x0 @ theta[nb:])[:, None]
            log_age = self._age_logprobs(eta0)
            loglik += float((initial_posterior * log_age).sum())
            centered = initial_posterior - initial_posterior.sum(-1, keepdims=True) * np.exp(log_age)
            # d(log survival at age d)/d(eta_j) = -h_j for d > j.
            tail_sum = np.flip(np.cumsum(np.flip(centered, axis=-1), axis=-1), axis=-1) - centered
            residual0 = -expit(eta0) * tail_sum
            # The aggregated terminal age has mass S(D-1)/h(D).
            residual0[..., -1] = -(1 - expit(eta0[..., -1])) * centered[..., -1]
            gradient[:nb] += basis.T @ residual0.sum(0)
            if q:
                gradient[nb:] += x0.T @ residual0.sum(-1)
        penalty = np.full(theta.size, self.ridge)
        penalty[0] = 1e-6
        return (-loglik + 0.5 * float((penalty * theta ** 2).sum()),
                -gradient + penalty * theta)

    def _m_step_hazard(self, w_leave, w_stay, xs, initial_posterior=None, x0=None):
        """Numerical generalized EM update including initial-age probabilities."""
        nb = self._basis().shape[1]
        q = xs.shape[-1] if self.hazard_predictors else 0
        for state in range(self.K):
            theta = np.r_[self.c_[state], self.beta_[state] if q else []]
            init = None if initial_posterior is None else initial_posterior[:, state]
            args = (state, w_leave[:, :, state], w_stay[:, :, state], xs, init, x0)
            result = minimize(self._hazard_objective, theta, args=args, method="L-BFGS-B", jac=True)
            if np.isfinite(result.fun) and result.fun <= self._hazard_objective(theta, *args)[0] + 1e-8:
                self.c_[state] = result.x[:nb]
                if q:
                    self.beta_[state] = result.x[nb:]

    def _parameter_names(self):
        return super()._parameter_names() + ("pi_", "c_", "beta_")

    def _penalty(self):
        off = ~np.eye(self.K, dtype=bool)
        return (super()._penalty() + 0.5 * self.ridge * float(
            (self.c_[:, 1:] ** 2).sum() + (self.beta_ ** 2).sum() +
            (self.A_[off] ** 2).sum() + (self.lam_ ** 2).sum()) +
            0.5e-6 * float((self.c_[:, 0] ** 2).sum()))

    def _observed_loglik(self, rain, x):
        self._x = x
        return self._fb(self._log_emission(rain, x), *self._pieces(x))[-1]

    def _m_step_dest(self, xi, xs):
        """Weighted multinomial logit for the destination of a state change."""
        K, q = self.K, xs.shape[-1]
        if K == 2:
            # There is a single possible destination per state: its probability
            # is one, so these unidentifiable logits must not retain a ridge cost.
            self.A_.fill(0.0)
            self.lam_.fill(0.0)
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
            base = NHMM(K, self.amounts, self.amount_predictor, self.thr, self.init_iter, self.tol, self.seed,
                        self.initial_predictors, self.initial_ridge, self.amount_resolution).fit(rain, x)
            for name in ("p_", "shape_", "scale_", "g_"):
                setattr(self, name, getattr(base, name).copy())
            self._x = x
            P = np.exp(base._trans(x))                                          # (Y,T,K,K)
            stay = np.einsum("ytk,ytkk->k", base.gamma_, P) / base.gamma_.sum((0, 1))
            self.A_, self.lam_ = base.A_.copy(), base.lam_.copy()
            self.pi_ = base.pi0_.copy()
            self.pi0_ = self.pi_.copy()
            self.initial_coef_ = base.initial_coef_.copy()
            self.nhmm_init_ = base
        else:
            self._init_emissions(rain, x)
            stay = np.full(K, 0.7)
            self.A_, self.lam_ = np.zeros((K, K)), np.zeros((K, q))
            self.pi_ = np.full(K, 1.0 / K)
            self.pi0_ = self.pi_.copy()
            self.initial_coef_ = np.zeros((q + 1, K - 1))
        self.c_ = np.zeros((K, nb))
        self.c_[:, 0] = np.log((1 - stay) / np.clip(stay, 1e-3, None))
        self.beta_ = np.zeros((K, q))
        xs = x[:, 1:]
        prev = -np.inf
        self.loglik_ = []
        self.objective_ = []
        for it in range(self.n_iter):
            self._x = x
            le = self._log_emission(rain)
            H, Q, init = self._pieces(x)
            a, b, c, e, ll = self._fb(le, H, Q, init)
            self.loglik_.append(ll)
            old_objective = ll - self._penalty()
            self.objective_.append(old_objective)
            previous = {name: getattr(self, name).copy() for name in self._parameter_names()}
            post = a * b                                                        # (Y,T,K,D)
            g = post.sum(-1)
            self.pi_ = np.clip(g[:, 0].mean(0), 1e-6, None); self.pi_ /= self.pi_.sum()
            self.pi0_ = self.pi_
            self._fit_initial(g[:, 0], x[:, 0])
            self._m_step_emissions(g, rain, x)
            w_leave, w_stay, xi = self._expected_counts(a, b, c, e, H, Q)
            self._m_step_hazard(w_leave, w_stay, xs, post[:, 0], x[:, 0])
            self._m_step_dest(xi, xs)
            self._accept_m_step(previous, rain, x, old_objective)
            if abs(ll - prev) < self.tol * abs(ll):
                break
            prev = ll
        self._x = x
        a, b, _, _, final_ll = self._fb(self._log_emission(rain), *self._pieces(x))
        post = a * b
        self.gamma_ = post.sum(-1)
        self.loglik_.append(final_ll)
        self.objective_.append(final_ll - self._penalty())
        self.duration_posterior_ = post.sum((0, 1))                              # expected days in (k, age)
        self.pi0_ = self.pi_
        return self

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

    Package alternative to ECC-Q of Roman Schefzik, Thordis L. Thorarinsdottir
    and Tilmann Gneiting (2013). Bin means preserve the finite sample mean
    exactly; point quantiles need not. This is not original ECC-Q.
    Reference: https://doi.org/10.1214/13-STS443.
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


def point_quantiles(sample, M, convention="midpoint"):
    """Marginal sample quantiles at explicitly chosen probability levels.

    ``convention='midpoint'`` uses (i - 1/2) / M (Scheuerer et al. 2017);
    ``convention='ecc'`` uses i / (M + 1) (Schefzik et al. 2013, ECC-Q).
    Uses linear interpolation on the finite empirical sample per column.
    Entirely missing columns remain missing; one-value columns stay constant.
    """
    x = np.asarray(sample, float)
    M = int(M)
    if x.ndim < 1 or x.shape[0] < 1 or M < 1:
        raise ValueError("sample must have a nonempty sample axis and M must be positive")
    if convention not in {"midpoint", "ecc"}:
        raise ValueError("convention must be 'midpoint' or 'ecc'")
    shape = x.shape[1:]
    ordered = np.sort(np.where(np.isfinite(x), x, np.nan).reshape(x.shape[0], -1), axis=0)
    count = np.isfinite(ordered).sum(0)
    u = ((np.arange(M, dtype=float) + 0.5) / M if convention == "midpoint" else
         np.arange(1, M + 1, dtype=float) / (M + 1))
    pos = u[:, None] * np.maximum(count[None] - 1, 0)
    lo = np.floor(pos).astype(int)
    hi = np.ceil(pos).astype(int)
    a = np.take_along_axis(ordered, lo, axis=0)
    b = np.take_along_axis(ordered, hi, axis=0)
    out = a + (b - a) * (pos - lo)
    return np.where(count[None] > 0, out, np.nan).reshape((M,) + shape)


def ensemble_copula_coupling(raw, calibrated, method="Q", rng=None):
    """ECC of Roman Schefzik, Thordis L. Thorarinsdottir and Tilmann Gneiting
    (2013), https://doi.org/10.1214/13-STS443.

    raw : (M, ...) raw ensemble supplying the rank (copula) structure
    calibrated : (N, ...) samples from the calibrated marginal of every column
        (N >= M; e.g. pooled corrected members of neighbouring days)
    method : 'Q' = M point quantiles at i/(M+1) (original ECC-Q),
        'Q-midpoint' = quantiles at (i-1/2)/M (CRPS quantization; 0.9.0 convention),
        'Q-mean' = M means of equal-probability bins (historical variant,
        preserves the marginal sample mean); 'R' = M random draws.
    Ties in the raw ensemble (dry days) are broken at random.
    Returns (M, ...) with the calibrated marginals and the raw ranks.
    ECC-Q/R restore the raw rank dependence; this assumes that the raw copula
    is useful. It does not independently correct an erroneous raw copula.
    """
    rng = rng if rng is not None else np.random.default_rng()
    raw = np.asarray(raw, float)
    cal = np.asarray(calibrated, float)
    if raw.ndim < 1 or cal.ndim != raw.ndim or raw.shape[1:] != cal.shape[1:] or raw.shape[0] < 1 or cal.shape[0] < 1:
        raise ValueError("raw and calibrated need nonempty member axes and identical column dimensions")
    M = raw.shape[0]
    if method == "Q":
        values = point_quantiles(cal, M, convention="ecc")
    elif method == "Q-midpoint":
        values = point_quantiles(cal, M, convention="midpoint")
    elif method == "Q-mean":
        values = stratified_quantiles(cal, M)
    elif method == "R":
        ordered = np.sort(np.where(np.isfinite(cal), cal, np.nan), axis=0)
        count = np.isfinite(ordered).sum(0)
        idx = (rng.random((M,) + cal.shape[1:]) * count[None]).astype(int)
        values = np.sort(np.take_along_axis(ordered, idx, axis=0), axis=0)
    else:
        raise ValueError("method must be 'Q', 'Q-midpoint', 'Q-mean' or 'R'")
    # An entirely missing raw column has no copula information. A partly
    # missing active column is rejected by schaake_shuffle rather than ranked
    # as artificial negative-infinite rainfall.
    values = np.where(np.isnan(raw).all(0)[None], np.nan, values)
    return schaake_shuffle(values, raw, axis=0, ties="random", rng=rng)


def preferential_dates(forecast_mean, obs_seasons, n, window=7, climatology=None):
    """Package forecast-mean analogue heuristic for a Schaake shuffle.

    Reordering comes from Martyn Clark, Subhrendu Gangopadhyay, Lauren Hay,
    Balaji Rajagopalan and Robert Wilby (2004); see module references.
    The forecast-mean ranking below is a package extension.

    forecast_mean : (T, S) forecast ensemble-mean (calibrated) field for each day
    obs_seasons   : (Y, T + 2*window, S) observed seasons padded by ``window`` days
    Each candidate trajectory (year y, shift o in [-window, window]) is scored by
    the mean squared difference of standardized anomalies (per site-day
    climatology of the candidates) between the forecast and the observed
    trajectory. The best shift of each year is taken first, so templates stay
    diverse; returns (year indices, offsets, scores) of the n best. Standard
    Schaake uses random candidates instead. This selects individual sequences
    by squared error to the forecast *mean*. It does not implement the minimum
    divergence Schaake shuffle (MDSS) of Scheuerer et al. (2017), which selects
    a set of sequences by agreement of all marginal forecast distributions.
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


def minimum_divergence_selection(calibrated, candidates, n, component_weights=None,
                                 temperature_forecast=None, temperature_candidates=None,
                                 temperature_pool=None):
    """Select a set of trajectories by Minimum Divergence Schaake Shuffle.

    Implements Eq. (5) of Michael Scheuerer, Thomas M. Hamill, Brett Whitin,
    Minxue He and Arthur Henkel (2017), https://doi.org/10.1002/2016WR020133,
    with exact one-at-a-time
    backward elimination. The predictive CDF is the finite empirical CDF of
    ``calibrated`` (sample, ...); ``candidates`` is (candidate, ...) on identical
    fields. The exact discrete objective is sum_j w_j integral (H_j-F_j)^2 dx.
    This is a numerical approximation to the article's continuous predictive
    CDFs; it is not forecast-mean analogue selection or a global optimum over
    all subsets. No duplicate candidates are generated when n exceeds support.

    Entirely missing predictive fields are ignored. Candidate trajectories
    must be complete on the active fields. ``component_weights`` has the field
    shape and defaults to one, as in the article's precipitation criterion.
    Weights may express fixed area/lead/variable scaling in mixed-unit uses.

    Optional matching temperature arrays apply the article's preliminary
    99-percent interval screen: retain all candidates with at most m violations,
    choosing the smallest m leaving at least ``temperature_pool`` (>= n) dates.
    The two variables then share the same selected trajectory indices.

    Returns selected original indices and a diagnostic dictionary containing
    objective history, deletion order, active fields and retained support.
    Memory is O(candidate_count * field_count), not O(candidate_count squared).
    Empirical marginals, optional component weights and seasonal windows are
    package adaptations. The paper's CSGD/EMOS marginal fits and river-basin
    streamflow evaluation are not reproduced here.
    """
    forecast = np.asarray(calibrated, float)
    historical = np.asarray(candidates, float)
    if (forecast.ndim < 2 or historical.ndim != forecast.ndim or
            forecast.shape[1:] != historical.shape[1:] or
            not forecast.shape[0] or not historical.shape[0]):
        raise ValueError("calibrated and candidates need nonempty sample axes and matching fields")
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    shape = forecast.shape[1:]
    weights = np.ones(shape) if component_weights is None else np.asarray(component_weights, float)
    if weights.shape != shape or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("component_weights must be finite nonnegative and match the field shape")
    flat_forecast = forecast.reshape(forecast.shape[0], -1)
    active = np.isfinite(flat_forecast).any(0) & (weights.ravel() > 0)
    if not active.any():
        raise ValueError("No finite predictive fields with positive component weight")
    flat = historical.reshape(historical.shape[0], -1)
    retained = np.flatnonzero(np.isfinite(flat[:, active]).all(1))
    if retained.size < n:
        raise ValueError("Not enough complete historical trajectories for n distinct MDSS templates")
    violations = None
    if (temperature_forecast is None) != (temperature_candidates is None):
        raise ValueError("temperature_forecast and temperature_candidates must be supplied together")
    if temperature_forecast is not None:
        tf, tc = np.asarray(temperature_forecast, float), np.asarray(temperature_candidates, float)
        if tf.ndim < 2 or tc.ndim != tf.ndim or tf.shape[1:] != tc.shape[1:] or tc.shape[0] != len(historical):
            raise ValueError("temperature arrays must have matching fields and the same candidate axis")
        target_pool = len(retained) if temperature_pool is None else int(temperature_pool)
        if target_pool < n:
            raise ValueError("temperature_pool must be at least n")
        af = np.isfinite(tf).any(0)
        if not af.any():
            raise ValueError("temperature forecast has no finite fields")
        interval = np.nanquantile(tf[:, af], [0.005, 0.995], axis=0)
        complete = np.isfinite(tc[retained][:, af]).all(1)
        retained = retained[complete]
        if len(retained) < n:
            raise ValueError("Not enough complete joint precipitation/temperature trajectories")
        values = tc[retained][:, af]
        violations = ((values < interval[0]) | (values > interval[1])).sum(1)
        k = min(target_pool, len(retained))
        cutoff = np.partition(violations, k - 1)[k - 1]
        retained = retained[violations <= cutoff]
    fields = flat[retained][:, active]
    forecast_fields = flat_forecast[:, active]
    weight = weights.ravel()[active]
    # Translating each marginal leaves all absolute distances unchanged and
    # reduces cancellation for large-offset variables (e.g. Kelvin temperature).
    center = np.nanmean(forecast_fields, axis=0)
    fields = fields - center
    forecast_fields = forecast_fields - center
    count = np.isfinite(forecast_fields).sum(0)
    # A_i = sum_j w_j E_F |historical_ij - Y_j|; variable finite sample counts
    # are respected instead of treating missing forecast values as zero.
    cross = np.zeros(len(fields))
    for sample in forecast_fields:
        valid = np.isfinite(sample)
        cross += (np.abs(fields[:, valid] - sample[valid]) * (weight[valid] / count[valid])).sum(1)

    def pair_row_sums(values):
        order = np.argsort(values, axis=0)
        ordered = np.take_along_axis(values, order, axis=0)
        prefix = np.cumsum(ordered, axis=0) - ordered
        rank = np.arange(len(values))[:, None]
        sums = (2 * rank - len(values)) * ordered - 2 * prefix + ordered.sum(0)
        inverse = np.argsort(order, axis=0)
        return (np.take_along_axis(sums, inverse, axis=0) * weight).sum(1)

    pair = pair_row_sums(fields)
    # The target-target constant is computed independently per marginal,
    # allowing missing members and preserving the empirical distribution.
    target_pair = 0.0
    for column in range(forecast_fields.shape[1]):
        sample = np.sort(forecast_fields[np.isfinite(forecast_fields[:, column]), column])
        ranks = np.arange(len(sample))
        target_pair += weight[column] * 2 * np.dot(2 * ranks - len(sample) + 1, sample) / len(sample) ** 2
    cross_sum, pair_sum = float(cross.sum()), float(pair.sum())
    current = np.arange(len(fields))
    divergence = lambda size: max(0.0, cross_sum / size - pair_sum / (2 * size ** 2) - target_pair / 2)
    history = [divergence(len(current))]
    removed = []
    while len(current) > n:
        size = len(current) - 1
        score = ((cross_sum - cross[current]) / size -
                 (pair_sum - 2 * pair[current]) / (2 * size ** 2) - target_pair / 2)
        drop = current[int(np.argmin(score))]
        removed.append(int(retained[drop]))
        current = current[current != drop]
        pair_sum -= 2 * pair[drop]
        cross_sum -= cross[drop]
        pair[current] -= (np.abs(fields[current] - fields[drop]) * weight).sum(1)
        history.append(divergence(len(current)))
    return retained[current], {"divergence": np.asarray(history), "removed_indices": np.asarray(removed, int),
                               "active_fields": active.reshape(shape), "retained_candidates": len(retained),
                               "target_cdf": "empirical", "algorithm": "one-at-a-time backward elimination"}


def minimum_divergence_dates(calibrated, obs_seasons, n, window=7, component_weights=None):
    """MDSS selection of year/offset templates from padded seasonal observations.

    Returns (year_indices, offsets, diagnostics); see
    ``minimum_divergence_selection`` for the exact distributional criterion
    of Michael Scheuerer, Thomas M. Hamill, Brett Whitin, Minxue He and Arthur
    Henkel (2017), https://doi.org/10.1002/2016WR020133. Padded seasonal
    candidates are a package adaptation of that criterion.
    """
    forecast, obs = np.asarray(calibrated, float), np.asarray(obs_seasons, float)
    w = int(window)
    if (forecast.ndim != 3 or obs.ndim != 3 or w < 0 or
            obs.shape[1:] != (forecast.shape[1] + 2 * w, forecast.shape[2]) or not len(obs)):
        raise ValueError("forecast (member,day,site) and padded observations (year,day+2*window,site) must agree")
    candidates = [(year, offset) for year in range(len(obs)) for offset in range(-w, w + 1)]
    trajectories = np.stack([obs[year, w + offset:w + offset + forecast.shape[1]] for year, offset in candidates])
    indices, info = minimum_divergence_selection(forecast, trajectories, n, component_weights)
    return (np.asarray([candidates[i][0] for i in indices]),
            np.asarray([candidates[i][1] for i in indices]), info)


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
    mean). ``"preferential"`` is a forecast-mean analogue selection, not MDSS
    as defined by Scheuerer et al. (2017). ``"mdss"`` implements backward
    elimination minimizing the full marginal-CDF divergence of historical
    trajectories against the empirical predictive sample. All shuffle methods
    use the same predictive point quantiles by default. Set
    ``ecc_quantiles="Q-midpoint"`` for the MDSS 2017 midpoint quantiles and the
    0.9.0 ECC convention, or ``ecc_quantiles="Q-mean"`` for
    the mean-preserving quantile-bin convention used in version 0.8.0.

    Scientific components: Schmidli, Frei and Vidale (2006), LOCI;
    Gudmundsson, Bremnes, Haugen and Engen-Skaugen (2012), QM review;
    Schefzik, Thorarinsdottir and Gneiting (2013), ECC;
    Clark, Gangopadhyay, Hay, Rajagopalan and Wilby (2004), Schaake reordering;
    Scheuerer, Hamill, Whitin, He and Henkel (2017), MDSS. Full references
    and DOI links are in the module docstring. Pooling and this composition
    are package choices, not one published experiment.
    """

    def __init__(self, months=(7, 8, 9), corrector="loci_qm", coupling="ecc", pool_days=2, window=7,
                 wet_threshold=1.0, seed=42, ecc_quantiles="Q"):
        if coupling not in {"none", "ecc", "schaake", "preferential", "mdss"}:
            raise ValueError("coupling must be 'none', 'ecc', 'schaake', 'preferential' or 'mdss'")
        if ecc_quantiles not in {"Q", "Q-midpoint", "Q-mean"}:
            raise ValueError("ecc_quantiles must be 'Q', 'Q-midpoint' or 'Q-mean'")
        self.months, self.coupling, self.pool, self.window = validate_months(months), coupling, int(pool_days), int(window)
        self.thr, self.seed, self.ecc_quantiles = float(wet_threshold), int(seed), ecc_quantiles
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
        templates (default: complete seasons in ``years``). ``downscale``
        only uses template seasons before the target season, unless
        ``allow_future_templates=True`` is explicitly passed. Fitting the
        corrector itself on a target year is still in-sample: for hindcast
        validation, refit on training years for each fold."""
        self.years_ = [int(y) for y in years]
        if not self.years_:
            raise ValueError("years must contain at least one complete calibration season")
        obs = canonicalize_observations(observations)["PRCP"]
        self.y_, self.x_ = obs["Y"].values, obs["X"].values
        mod = season_blocks(self._model_on_grid(hindcast), self.years_, self.months)  # (Y, M, T, S)
        ob = season_blocks(obs, self.years_, self.months)[:, 0]                 # (Y, T, S)
        self.month_ = season_dates(self.years_[0], self.months).month.to_numpy()
        self.corrector.fit(mod, ob, self.month_)
        # Default templates are limited to fit years, rather than any future
        # observed season that happens to be present in the supplied dataset.
        t_obs = pd.DatetimeIndex(obs["T"].values)
        candidates = self.years_ if template_years is None else [int(y) for y in template_years]
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

    def downscale(self, forecast: xr.DataArray, year, n_members=None, allow_future_templates=False):
        """Daily forecast on the fitted grid for season ``year``.

        ``allow_future_templates=True`` permits noncausal historical templates
        for an explicit sensitivity experiment; avoid it in hindcasts.
        """
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
                out = ensemble_copula_coupling(raw[:M], pooled, self.ecc_quantiles, rng)
            else:
                values = (stratified_quantiles(pooled, M) if self.ecc_quantiles == "Q-mean" else
                          point_quantiles(pooled, M, convention=("ecc" if self.ecc_quantiles == "Q" else "midpoint")))
                keep = np.array([y != int(year) and (allow_future_templates or y < int(year))
                                 for y in self.obs_years_])
                pool_idx = np.flatnonzero(keep)
                if pool_idx.size == 0:
                    raise ValueError("No eligible observed template seasons precede the target year; "
                                     "fit with earlier template_years or explicitly set allow_future_templates=True")
                if self.coupling == "schaake":
                    C = pool_idx.size * (2 * self.window + 1)
                    pick = rng.choice(C, M, replace=M > C)
                    yi, oi = pool_idx[pick // (2 * self.window + 1)], pick % (2 * self.window + 1) - self.window
                elif self.coupling == "mdss":
                    yi, oi, selection = minimum_divergence_dates(pooled, self.obs_padded_[keep], M, self.window)
                    yi = pool_idx[yi]
                    info["mdss"] = selection
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
        if self.coupling == "ecc":
            ds.attrs["ecc_quantiles"] = self.ecc_quantiles
        if self.coupling != "none":
            ds.attrs["marginal_quantiles"] = self.ecc_quantiles
        if self.coupling == "mdss":
            ds.attrs["mdss_target_cdf"] = "empirical"
            ds.attrs["mdss_selection"] = "one-at-a-time backward elimination"
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
           "preferential_dates", "minimum_divergence_selection", "minimum_divergence_dates", "DynamicalDownscaler", "synthetic_model_ensemble", "season_blocks"]
