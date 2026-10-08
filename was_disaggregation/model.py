"""High-level xarray interface for the gridded parametric weather generator."""
from __future__ import annotations
import json
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from .data import canonicalize_observations, prepare_probabilities, seasonal_cube, season_dates, validate_months
from .conditioning import (classify_seasons, year_weights, normal_scores,
                           tercile_pdf_ratio_weights, class_weights)
from scipy.special import ndtr
from .rainfall import fit_rainfall, simulate_rainfall, seasonal_total_moments
from .multivariate import fit_multivariate, simulate_multivariate
from .spatial import GaussianSpatialField, IndependentField, fit_distance_model, _integer


class WeatherGenerator:
    """Condition daily weather on seasonal PRCP tercile probabilities.

    Other variables respond through their historical association with PRCP
    categories. They are not independently conditioned on temperature forecasts.
    The baseline MUST match the forecast provider; 1991–2020 is an explicit
    configurable assumption, since the supplied forecast has no baseline metadata.
    Use ``generate_dask`` for domains larger than ``max_sites``.

    Options added in 0.2.0
    ----------------------
    conditioning : {'mean', 'mixture'}
        'mean' is Wilks (2002): one parameter set equal to the forecast-weighted
        average over tercile classes (Eqs. 2, 4-8). Averaging collapses the
        three-class mixture, so seasonal totals are under-dispersed and the
        near-normal category is over-produced. 'mixture' fits one parameter set
        per tercile class (years of that class only) and lets every member draw
        its class from (PB, PN, PA). This retains variability between the
        class-specific parameter sets, subject to the shrinkage below.
    mixture_shrinkage : {'variance', 'none'} or float in [0, 1]
        Tercile classes are defined by the outcome itself, so class-fitted
        parameters also absorb the weather noise that selected those years; a
        raw mixture ('none') over-disperses seasonal totals. Member weights are
        blended as (1-k) w_forecast + k w_class; 'variance' chooses k per cell so
        that, under climatological probabilities, approximate stationary
        Markov moments (Katz 1985) approach the observed interannual variance.
        This is a heuristic: month boundaries, finite-season corrections and
        semi-Markov run hazards are not represented by those moments. Nonlinear
        parameter fitting also prevents an exact forecast-mean identity.
    class_draw : {'fitted', 'shared'}
        How a member's class varies in space. 'shared' uses one uniform per
        member for the whole domain (class = forecast quantile at every cell);
        'fitted' uses a Gaussian field whose range is fitted to the spatial
        correlation of historical seasonal-total normal scores.
    weighting : {'tercile', 'pdf_ratio', 'mre', 'croley'}
        'mre' / 'croley' (0.3.0): several tercile forecasts at once (total, onset,
        dry spells, cessation ...) via ``constraints``; weights minimise the
        relative entropy (or Croley's squared distortion) from ``year_prior``.
    constraints : list of :class:`was_disaggregation.mre.SeasonalConstraint`
        Used with 'mre'/'croley'. A constraint with ``probabilities=None`` uses
        the main ``probabilities`` passed to ``fit`` (e.g. a JAS total while the
        generator runs May-October). If none does, the seasonal total over
        ``months`` is added automatically with the main forecast.
    occurrence : {'markov', 'spell'}
        'spell' (0.4.0): semi-Markov wet/dry sequence whose transition
        probabilities depend on the length of the current dry/wet run, fitted
        from forecast-weighted historical runs. Needed for dry-spell forecasts:
        a first-order chain cannot reproduce long dry spells. ``max_dry_run``,
        ``max_wet_run`` cap the run-length classes; ``spell_prior`` shrinks
        sparse classes towards the first-order estimate.
    year_prior : DataArray (season_year), optional
        Prior year weights, e.g. :func:`enso_rank_weights`.
        'tercile' = p(category)/N(category) (Briggs & Wilks 1996). 'pdf_ratio' =
        Stedinger & Kim (2010) with a normal-score forecast density matched to
        the terciles and calibrated to keep exact category masses.
    amount_distribution : {'gamma', 'mixed_exponential'}
        'mixed_exponential' follows Wilks (2002); with ``hold_alpha`` the mixing
        weight is the climatological one and only the two means are conditioned.
    trace_rainfall : bool
        Simulate sub-threshold rain on non-wet days (default True). Tercile
        thresholds come from full observed totals; dropping drizzle biases the
        generated totals low and inflates the below-normal frequency.
    persistence : {'climatology', 'weighted', 'independent'}
        'weighted' re-estimates d from forecast-weighted transitions (Eq. 3 analogue).
    """
    def __init__(self, months=(7, 8, 9), climatology=(1991, 2020),
                 wet_threshold=1., spatial="distance", seed=42,
                 n_features=128, spatial_models=None, max_pairs=300,
                 max_spatial_sites=64, max_sites=2500,
                 tercile_method="empirical", empty_policy="climatology",
                 train_years=None, persistence="climatology",
                 conditioning="mean", class_draw="fitted", weighting="tercile",
                 amount_distribution="gamma", hold_alpha=True, mixture_shrinkage="variance",
                 trace_rainfall=True, constraints=None, year_prior=None, total_tolerance=0.01,
                 occurrence="markov", max_dry_run=40, max_wet_run=15, spell_prior=3.0):
        if occurrence not in {"markov", "spell"}:
            raise ValueError("occurrence must be 'markov' or 'spell'")
        self.occurrence = occurrence
        self.max_dry_run = _integer(max_dry_run, "max_dry_run", 1)
        self.max_wet_run = _integer(max_wet_run, "max_wet_run", 1)
        self.spell_prior = float(spell_prior)
        if not np.isfinite(self.spell_prior) or self.spell_prior < 0:
            raise ValueError("spell_prior must be finite and nonnegative")
        if conditioning not in {"mean", "mixture"}:
            raise ValueError("conditioning must be 'mean' or 'mixture'")
        if class_draw not in {"fitted", "shared"}:
            raise ValueError("class_draw must be 'fitted' or 'shared'")
        if weighting not in {"tercile", "pdf_ratio", "mre", "croley"}:
            raise ValueError("weighting must be 'tercile', 'pdf_ratio', 'mre' or 'croley'")
        self.constraints = list(constraints or [])
        self.year_prior = year_prior
        if year_prior is not None and (not isinstance(year_prior, xr.DataArray) or year_prior.dims != ("season_year",)):
            raise ValueError("year_prior must be a DataArray with dimension season_year")
        self.total_tolerance = float(total_tolerance)
        if self.constraints and weighting not in {"mre", "croley"}:
            raise ValueError("constraints require weighting='mre' or 'croley'")
        if year_prior is not None and weighting not in {"mre", "croley"}:
            raise ValueError("year_prior requires weighting='mre' or 'croley'")
        if amount_distribution not in {"gamma", "mixed_exponential"}:
            raise ValueError("amount_distribution must be 'gamma' or 'mixed_exponential'")
        self.conditioning, self.class_draw, self.weighting = conditioning, class_draw, weighting
        self.amount_distribution, self.hold_alpha = amount_distribution, bool(hold_alpha)
        if not ((isinstance(mixture_shrinkage, str) and mixture_shrinkage in {"variance", "none"}) or
                (isinstance(mixture_shrinkage, (int, float)) and 0 <= mixture_shrinkage <= 1)):
            raise ValueError("mixture_shrinkage must be 'variance', 'none' or a number in [0, 1]")
        self.mixture_shrinkage = mixture_shrinkage
        self.trace_rainfall = bool(trace_rainfall)
        if spatial not in {"distance", "independent"}:
            raise ValueError("spatial must be 'distance' or 'independent'")
        if not np.isfinite(wet_threshold) or wet_threshold <= 0:
            raise ValueError("wet_threshold must be positive")
        self.months = validate_months(months)
        self.climatology = tuple(climatology)
        self.wet_threshold = float(wet_threshold)
        self.spatial = spatial
        self.seed = _integer(seed, "seed")
        self.n_features = _integer(n_features, "n_features", 1)
        self.spatial_models = spatial_models
        self.max_pairs = _integer(max_pairs, "max_pairs", 1)
        self.max_spatial_sites = _integer(max_spatial_sites, "max_spatial_sites", 1)
        self.max_sites = _integer(max_sites, "max_sites", 1)
        self.tercile_method = tercile_method
        self.empty_policy = empty_policy
        self.train_years = train_years
        self.persistence = persistence

    def fit(self, observations: xr.Dataset, probabilities: xr.DataArray, *, site_ids=None):
        """Fit a bounded spatial domain; no array is interpolated silently in time."""
        obs = canonicalize_observations(observations)
        if "PRCP" not in obs:
            raise ValueError("PRCP observations are required")
        nsite = obs.sizes["Y"] * obs.sizes["X"]
        if nsite > self.max_sites:
            raise ValueError(f"{nsite} cells exceed max_sites={self.max_sites}; use generate_dask")
        self.probabilities_ = prepare_probabilities(probabilities, target=obs)
        cube = seasonal_cube(obs, months=self.months)
        if self.train_years is not None:
            wanted = np.asarray(self.train_years, dtype=int)
            cube = cube.sel(season_year=cube.season_year.isin(wanted))
        if cube.sizes.get("season_year", 0) < 3:
            raise ValueError("At least three complete historical seasons are required")
        # Deliberate bounded materialization. The scalable interface only calls
        # this method on tiles and shares global spatial-field parameters.
        cube = cube.load()
        self.y_, self.x_ = obs.Y.values, obs.X.values
        self.units_ = {v: obs[v].attrs.get("units", "") for v in obs.data_vars}
        self.years_ = cube.season_year.values.astype(int)
        self.month_ = cube.month.values.astype(int)
        n_year, n_day = cube.sizes["season_year"], cube.sizes["day"]
        values = {v: cube[v].transpose("season_year", "day", "Y", "X").values.reshape(n_year, n_day, nsite)
                  for v in cube.data_vars}
        rain = values.pop("PRCP")
        # A missing day must never become a reduced/zero seasonal total.
        totals = np.where(np.isfinite(rain).all(axis=1), np.nansum(rain, axis=1), np.nan)
        cats, thresholds = classify_seasons(totals, self.years_, climatology=self.climatology,
                                            method=self.tercile_method)
        prob = self.probabilities_.transpose("probability", "Y", "X").values.reshape(3, nsite)
        weights, flags = year_weights(cats, prob, empty_policy=self.empty_policy)
        if self.weighting == "pdf_ratio":
            zscores = normal_scores(totals, self.years_, climatology=self.climatology,
                                    method=self.tercile_method)
            pdf_w, _ = tercile_pdf_ratio_weights(zscores, cats, prob, empty_policy=self.empty_policy)
            weights = np.where(weights.sum(axis=0, keepdims=True) > 0, pdf_w, 0.)
        elif self.weighting in {"mre", "croley"}:
            weights, flags, cats, thresholds = self._constrained_weights(obs, totals, prob, nsite)
            zscores = None
        else:
            zscores = None
        self.prob_flat_ = prob
        implied = np.stack([(weights * (cats == c)).sum(axis=0) for c in range(3)])
        self.thresholds_ = xr.DataArray(thresholds.reshape(2, len(self.y_), len(self.x_)),
            dims=("quantile", "Y", "X"), coords={"quantile": [1/3, 2/3], "Y": self.y_, "X": self.x_},
            name="seasonal_PRCP_threshold", attrs={"units": "mm", "climatology": str(self.climatology)})
        self.year_weights_ = xr.DataArray(weights.reshape(n_year, len(self.y_), len(self.x_)),
            dims=("season_year", "Y", "X"), coords={"season_year":self.years_, "Y":self.y_, "X":self.x_})
        rain_options = dict(wet_threshold=self.wet_threshold, persistence=self.persistence,
                            amount_distribution=self.amount_distribution, include_trace=self.trace_rainfall)
        if self.occurrence == "spell":
            rain_options.update(occurrence="spell", max_dry_run=self.max_dry_run, max_wet_run=self.max_wet_run,
                                spell_prior=self.spell_prior,
                                initial_state=self._initial_state(obs, nsite))
        if self.amount_distribution == "mixed_exponential":
            # Climatological mixed exponential: mixing weight held fixed (Wilks 2002)
            # and starting values for every conditioned EM fit.
            clim_w = (cats >= 0).astype(float)
            clim = fit_rainfall(rain, self.month_, clim_w, **rain_options)
            rain_options["amount_init"] = (clim.alpha, clim.beta1, clim.beta2)
            if self.hold_alpha:
                rain_options["alpha_fixed"] = clim.alpha
        self.rain_fit_ = fit_rainfall(rain, self.month_, weights, **rain_options)
        self.multi_fit_ = fit_multivariate(values, rain, self.month_, weights,
                                           wet_threshold=self.wet_threshold) if values else None
        self.class_rain_fits_, self.class_multi_fits_ = None, None
        if self.conditioning == "mixture":
            cw, empty_class = class_weights(weights, cats, fallback_weights=weights)
            kappa = self._mixture_kappa(rain, cw, cats, totals, rain_options, nsite)
            cw = (1 - kappa)[None, None, :] * weights[None] + kappa[None, None, :] * cw
            self.mixture_kappa_ = xr.DataArray(kappa.reshape(len(self.y_), len(self.x_)), dims=("Y", "X"),
                coords={"Y": self.y_, "X": self.x_}, name="mixture_kappa",
                attrs={"description": "share of tercile-class deviation kept in member parameters"})
            self.class_rain_fits_ = [fit_rainfall(rain, self.month_, cw[c], **rain_options) for c in range(3)]
            if self.multi_fit_ is not None:
                self.class_multi_fits_ = [fit_multivariate(values, rain, self.month_, cw[c],
                                                           wet_threshold=self.wet_threshold,
                                                           dependence=self.multi_fit_) for c in range(3)]
            self.empty_class_ = empty_class
        lat, lon = np.meshgrid(self.y_, self.x_, indexing="ij")
        self.lat_, self.lon_ = lat.ravel(), lon.ravel()
        self.site_ids_ = (np.arange(nsite, dtype=np.uint64) if site_ids is None
                          else np.asarray(site_ids).reshape(-1))
        if self.site_ids_.size != nsite:
            raise ValueError("site_ids must contain one ID per grid cell")
        self.spatial_models_ = dict(self.spatial_models or {})
        stream_values = {"occurrence": np.where(np.isfinite(rain), (rain >= self.wet_threshold).astype(float), np.nan),
                         "amount": np.where(rain > self.wet_threshold, rain-self.wet_threshold, np.nan)}
        self.variable_names_ = list(self.multi_fit_.variables) if self.multi_fit_ is not None else []
        if self.multi_fit_ is not None:
            residuals = getattr(self.multi_fit_, "climatological_innovations", None)
            if residuals is None:
                residuals = self.multi_fit_.climatological_residuals
            stream_values.update(residuals)
        if self.conditioning == "mixture" and self.class_draw == "fitted":
            # Seasonal anomalies (years as samples) define how far a member's
            # tercile class stays coherent in space.
            z = zscores if zscores is not None else normal_scores(
                totals, self.years_, climatology=self.climatology, method=self.tercile_method)
            stream_values["season_class"] = z
        if self.spatial == "distance":
            # Evenly spaced grid IDs avoid dense pairwise covariance matrices.
            sample = np.unique(np.linspace(0, nsite-1, min(nsite, self.max_spatial_sites)).astype(int))
            for key, arr in stream_values.items():
                if key in self.spatial_models_:
                    continue
                data = np.asarray(arr).reshape(-1, nsite)[:, sample]
                self.spatial_models_[key] = fit_distance_model(data, self.lat_[sample], self.lon_[sample],
                    kind="power" if key == "occurrence" else "exponential", max_pairs=self.max_pairs,
                    seed=self.seed, transform="binary" if key == "occurrence" else "gaussian",
                    **({"min_overlap": 10} if key == "season_class" else {}))
        self.diagnostics_ = {
            "training_years": self.years_.tolist(), "climatology": list(self.climatology),
            "climatology_assumption": "Must be confirmed against forecast producer metadata",
            "n_sites": nsite, "n_days": n_day,
            "weight_flags_counts": {str(k):int((flags == k).sum()) for k in np.unique(flags)},
            "weight_flags": (self.constraint_info_.attrs["flags"] if self.weighting in {"mre", "croley"}
                             else "bits: 1 empty-category fallback; 2 invalid forecast; 4 no complete history"),
            "conditioning": self.conditioning, "weighting": self.weighting,
            "amount_distribution": self.amount_distribution,
            "weight_implied_probability_mean": np.nanmean(np.where(prob >= 0, implied, np.nan), axis=1).tolist(),
            "rainfall": self.rain_fit_.diagnostics,
            "other_variables": getattr(self.multi_fit_, "diagnostics", {}),
            "spatial": "whole-season stationary kernels; approximate Fourier Gaussian copula" if self.spatial == "distance" else "independent site innovations",
            "forecast_match": "Parameter conditioning does not guarantee exact seasonal category frequencies"}
        self.weight_implied_probability_ = xr.DataArray(implied.reshape(3, len(self.y_), len(self.x_)),
            dims=("probability", "Y", "X"), coords={"probability": ["PB", "PN", "PA"], "Y": self.y_, "X": self.x_})
        if self.conditioning == "mixture":
            self.diagnostics_["empty_class_cells"] = self.empty_class_.sum(axis=1).tolist()
            self.diagnostics_["mixture_kappa_mean"] = float(np.nanmean(self.mixture_kappa_.values))
        if np.any(flags & 1):
            warnings.warn("Some cells have empty forecast categories; inspect diagnostics_ weight flags", UserWarning)
        return self

    def generate(self, year: int, n_members=20, *, member_start=0, allow_historical=False):
        """Generate one season; year is the year of its first month.

        ``member_start`` makes separate member batches reproducible. Historical
        target years require explicit opt-in; independent hindcasts need refitting.
        """
        if not hasattr(self, "rain_fit_"):
            raise RuntimeError("Call fit before generate")
        if (isinstance(n_members, (bool, np.bool_)) or not isinstance(n_members, (int, np.integer))
                or isinstance(member_start, (bool, np.bool_)) or not isinstance(member_start, (int, np.integer))
                or n_members < 1 or member_start < 0):
            raise ValueError("n_members must be positive and member_start nonnegative")
        dates = season_dates(year, self.months)
        if not allow_historical and year <= int(self.years_.max()):
            raise ValueError("Target year overlaps training history; refit on earlier years or explicitly allow_historical=True for scenarios")
        if len(dates) != len(self.month_):
            raise ValueError("Historical and generated season lengths differ")
        ns = len(self.lat_)
        bytes_needed = n_members * len(dates) * ns * (1 + len(self.variable_names_)) * 8
        if bytes_needed > 2_000_000_000:
            raise MemoryError("Generation exceeds 2 GB output working array; use generate_dask with member batches")
        fields = {}
        stream_keys = {0:"occurrence", 1:"amount", 99:"season_class",
                       **{10+i:v for i,v in enumerate(self.variable_names_)}}
        def draw(n, step, stream):
            if stream not in fields:
                if self.spatial == "independent":
                    fields[stream] = IndependentField(self.lat_, self.lon_, seed=self.seed, site_ids=self.site_ids_)
                else:
                    key = stream_keys.get(stream)
                    if key not in self.spatial_models_:
                        raise KeyError(f"Missing spatial model for stream {stream} ({key})")
                    fields[stream] = GaussianSpatialField(self.lat_, self.lon_, model=self.spatial_models_[key],
                        seed=self.seed, n_features=self.n_features, site_ids=self.site_ids_)
            return fields[stream].sample(n+member_start, step, stream)[member_start:]
        rain_fit, multi_fit, classes = self.rain_fit_, self.multi_fit_, None
        if self.conditioning == "mixture":
            classes = self._draw_classes(n_members, member_start, draw)
            b, n, a = self.class_rain_fits_
            rain_fit = b.select(classes, (n, a))
            rain_fit.valid = rain_fit.valid & self.rain_fit_.valid
            if self.class_multi_fits_ is not None:
                mb, mn, ma = self.class_multi_fits_
                multi_fit = mb.select(classes, (mn, ma))
                multi_fit.valid = multi_fit.valid & self.multi_fit_.valid
        rain = simulate_rainfall(rain_fit, self.month_, n_members, draw, wet_threshold=self.wet_threshold)
        generated = {"PRCP": rain}
        if multi_fit is not None:
            generated.update(simulate_multivariate(multi_fit, rain, self.month_, draw,
                                                  wet_threshold=self.wet_threshold))
            self.diagnostics_["simulation_constraints"] = multi_fit.simulation_diagnostics
        ds = xr.Dataset({v: (("member", "T", "Y", "X"), a.reshape(n_members, len(dates), len(self.y_), len(self.x_)).astype("float32"))
                         for v,a in generated.items()},
                        coords={"member":np.arange(member_start,member_start+n_members), "T":dates, "Y":self.y_, "X":self.x_})
        for v in ds:
            ds[v].attrs["units"] = "mm d-1" if v == "PRCP" else self.units_.get(v, "")
        if classes is not None:
            ds["tercile_class"] = (("member", "Y", "X"), np.where(rain_fit.valid, classes, -1)
                                   .reshape(n_members, len(self.y_), len(self.x_)).astype("int8"))
            ds["tercile_class"].attrs.update(flag_values="0 1 2", flag_meanings="below near above",
                                             description="tercile class whose parameters drove each member/cell")
        from ._version import __version__
        ds.attrs.update(generator=f"was-disaggregation {__version__}: forecast-conditioned Richardson/Wilks extension",
                        conditioning=self.conditioning, weighting=self.weighting,
                        amount_distribution=self.amount_distribution, occurrence=self.occurrence,
                        season_months=",".join(map(str,self.months)), seed=self.seed,
                        climatology=f"{self.climatology[0]}-{self.climatology[1]}",
                        training_years=f"{self.years_.min()}-{self.years_.max()}",
                        wet_threshold_mm=self.wet_threshold, spatial_method=self.spatial,
                        caveat="Daily parameter conditioning; evaluate seasonal tercile frequency mismatch",
                        leap_day_policy="February 29 excluded")
        return ds

    def _initial_state(self, obs, nsite):
        """Wet/dry state and run length on the day before each season (year, site).

        Looks back max(max_dry_run, max_wet_run)+1 days in the full daily record,
        so runs in progress at season start (e.g. the dry season) are known.
        """
        look = max(self.max_dry_run, self.max_wet_run) + 1
        stacked = obs["PRCP"].stack(site=("Y", "X")).transpose("T", "site")
        wet0 = np.full((len(self.years_), nsite), np.nan)
        run0 = np.full((len(self.years_), nsite), np.nan)
        for i, year in enumerate(self.years_):
            start = season_dates(int(year), self.months)[0]
            dates = pd.date_range(start - pd.Timedelta(days=look + 1), start - pd.Timedelta(days=1))
            dates = dates[~((dates.month == 2) & (dates.day == 29))][-look:]
            x = np.asarray(stacked.reindex(T=dates).values, dtype=float)[::-1]    # day before start first
            ok = np.isfinite(x)
            state = x >= self.wet_threshold
            same = ok & (state == state[0][None])
            run = np.argmin(np.vstack([same, np.zeros((1, nsite), bool)]), axis=0)
            wet0[i] = np.where(ok[0], state[0].astype(float), np.nan)
            run0[i] = np.where(ok[0], run, np.nan)
        return wet0, run0

    def _constrained_weights(self, obs, totals, prob, nsite):
        """Minimum relative entropy / Croley weights for all constraints."""
        from .mre import constrained_year_weights, relabel_probabilities, thresholds_for
        n_year = len(self.years_)
        values, probs, tols, primary, thr_fixed = {}, {}, {}, None, {}
        for con in self.constraints:
            if con.name in values:
                raise ValueError(f"duplicate constraint name {con.name!r}")
            if con.probabilities is None:
                if primary is not None:
                    raise ValueError("only one constraint can use the main forecast (probabilities=None)")
                p, primary = prob, con.name
            else:
                p = prepare_probabilities(relabel_probabilities(con.probabilities), target=obs)
                p = p.transpose("probability", "Y", "X").values.reshape(3, nsite)
            v = con.attribute.compute(obs["PRCP"], self.years_)
            values[con.name] = v.transpose("season_year", "Y", "X").values.reshape(n_year, nsite)
            probs[con.name], tols[con.name] = p, con.tolerance
            fixed = thresholds_for(con, target=obs, n_sites=nsite)
            if fixed is not None:
                thr_fixed[con.name] = fixed
        if primary is None:
            primary = "total" if "total" not in values else "forecast_total"
            values[primary], probs[primary], tols[primary] = totals, prob, self.total_tolerance
        prior = None
        if self.year_prior is not None:
            prior = np.asarray(self.year_prior.sel(season_year=self.years_).values, dtype=float)
        weights, flags, info = constrained_year_weights(
            values, probs, self.years_, climatology=self.climatology, prior=prior, tolerances=tols,
            method=self.weighting, tercile_method=self.tercile_method, thresholds=thr_fixed)
        shape = (len(self.y_), len(self.x_))
        coords = {"constraint": info["names"], "probability": ["PB", "PN", "PA"], "Y": self.y_, "X": self.x_}
        self.constraint_info_ = xr.Dataset({
            "target_probability": (("constraint", "probability", "Y", "X"),
                                   np.stack([info["target_probability"][k].reshape(3, *shape) for k in info["names"]])),
            "achieved_probability": (("constraint", "probability", "Y", "X"),
                                     np.stack([info["achieved_probability"][k].reshape(3, *shape) for k in info["names"]])),
            "effective_years": (("Y", "X"), info["effective_years"].reshape(shape)),
            "kl_from_prior": (("Y", "X"), info["kl_from_prior"].reshape(shape)),
            "flags": (("Y", "X"), flags.reshape(shape)),
        }, coords=coords, attrs={"method": self.weighting, "primary_constraint": primary,
                                 "flags": "2 invalid forecast; 4 no history; 8 constraint missed by >0.05; 16 <5 effective years"})
        self.constraint_attributes_ = {k: xr.DataArray(v.reshape(n_year, *shape), dims=("season_year", "Y", "X"),
                                                       coords={"season_year": self.years_, "Y": self.y_, "X": self.x_}, name=k)
                                       for k, v in values.items()}
        if np.any(flags & 8):
            warnings.warn("Some cells cannot meet all constraints within 0.05; inspect constraint_info_", UserWarning)
        return weights, flags, info["categories"][primary], info["thresholds"][primary]

    def _mixture_kappa(self, rain, cw, cats, totals, rain_options, nsite):
        """Per-cell blend factor k in [0, 1] (see ``mixture_shrinkage``)."""
        if self.mixture_shrinkage == "none":
            return np.ones(nsite)
        if not isinstance(self.mixture_shrinkage, str):
            return np.full(nsite, float(self.mixture_shrinkage))
        moments = [seasonal_total_moments(fit_rainfall(rain, self.month_, cw[c], **rain_options), self.month_)
                   for c in range(3)]
        means = np.stack([m for m, _ in moments])
        noise = np.nanmean(np.stack([v for _, v in moments]), axis=0)
        between = np.nanvar(means, axis=0)
        reference = (self.years_ >= self.climatology[0]) & (self.years_ <= self.climatology[1])
        observed = np.nanvar(np.where(cats[reference] >= 0, totals[reference], np.nan), axis=0, ddof=1)
        kappa2 = np.divide(observed - noise, between, out=np.zeros(nsite), where=between > 0)
        kappa = np.sqrt(np.clip(np.nan_to_num(kappa2), 0, 1))
        self._kappa_moments = {"noise_var": noise, "between_var": between, "observed_var": observed}
        return kappa

    def _draw_classes(self, n_members, member_start, draw):
        """Tercile class (member, site) for mixture conditioning.

        Uniforms are reproducible per member index (tile/batch invariant):
        'shared' uses one per member, 'fitted' a spatial Gaussian field.
        """
        if self.class_draw == "fitted":
            u = ndtr(draw(n_members, -2, 99))
        else:
            idx = np.arange(member_start, member_start + n_members)
            u = np.array([np.random.default_rng(np.random.SeedSequence([self.seed, 90901, int(m)])).random()
                          for m in idx])[:, None] * np.ones((1, len(self.lat_)))
        p = np.nan_to_num(self.prob_flat_, nan=1. / 3.)
        return ((u > p[0][None]).astype(np.int8) + (u > (p[0] + p[1])[None]).astype(np.int8))

    def save_diagnostics(self, path):
        """Write fit diagnostics without serializing executable Python objects."""
        def convert(o):
            if isinstance(o,np.ndarray): return o.tolist()
            if isinstance(o,np.generic): return o.item()
            return str(o)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"fit":self.diagnostics_, "spatial_models":self.spatial_models_},f,indent=2,default=convert)
