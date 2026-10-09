"""Coherent whole-season analogs and marginal-preserving rank reordering.

The whole-season analog is a transparent baseline, not the Apipattanavis KNN
algorithm. A donor is shared by every variable, day and grid cell. This preserves
the historical weather field but a single regional donor distribution generally
cannot satisfy a different tercile forecast at every cell.

Scientific scope
----------------
``schaake_shuffle`` implements the rank-permutation core of Clark et al.
(2004b). ``ForecastSchaakeGenerator`` is a forecast-conditioned variant of
Clark et al. (2004a): templates are complete seasons aligned to the target
calendar, without an initial random date offset. Daily marginal donors do use
the configured date window. ``enso_rank_weights`` integrates Clark's Eq. (9)
rank rule exactly, but does not fit its parameters by RPSS or reproduce the
complete Yates regional kNN scenario generator. Weighted templates, physical
bound swaps and missing-data policies are declared package extensions.

References
----------
* Martyn P. Clark, Subhrendu Gangopadhyay, David Brandon, Kevin Werner,
  Lauren E. Hay, Balaji Rajagopalan and David Yates (2004a), "A Resampling
  Procedure for Generating Conditioned Daily Weather Sequences".
  https://doi.org/10.1029/2003WR002747
  Core: conditional marginal resampling, coherent historical rank templates,
  and climate-index rank selection; calendar-aligned templates are a variant.
* Martyn P. Clark, Subhrendu Gangopadhyay, Lauren E. Hay, Balaji Rajagopalan
  and Robert L. Wilby (2004b), "The Schaake Shuffle: A Method for Reconstructing
  Space-Time Variability in Forecasted Precipitation and Temperature Fields".
  https://doi.org/10.1175/1525-7541(2004)005<0243:TSSAMF>2.0.CO;2
  Core: rank reordering; random tie breaking is explicit and does not promise
  exact Spearman agreement when historical values are tied.
* David Yates, Subhrendu Gangopadhyay, Balaji Rajagopalan and Kenneth Strzepek
  (2003), "A Technique for Generating Regional Climate Scenarios Using a
  Nearest-Neighbor Algorithm". https://doi.org/10.1029/2002WR001769
  Background: preferential rank selection only; the full generator is absent.
* William M. Briggs and Daniel S. Wilks (1996), "Extension of the Climate
  Prediction Center Long-Lead Temperature and Precipitation Outlooks to General
  Weather Statistics". https://doi.org/10.1175/1520-0442(1996)009<3496:EOTCPC>2.0.CO;2
  Core component: forecast-conditioned historical-year category masses.
* Jery R. Stedinger and Young-Oh Kim (2010), "Probabilities for Ensemble
  Forecasts Reflecting Climate Information".
  https://doi.org/10.1016/j.jhydrol.2010.06.038
  Core component: density-ratio weights; normal-score density and optional
  category recalibration are package variants, defined in conditioning.py.

Scientific references identify method sources, not software authorship, which
is recorded separately in package metadata and copyright notices.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from .conditioning import classify_seasons, year_weights, normal_scores, tercile_pdf_ratio_weights
from .data import prepare_probabilities, seasonal_cube, season_dates


def schaake_shuffle(ensemble, template, axis: int = 0, ties: str = "stable", rng=None):
    """Reorder ensemble values to the ranks of a same-sized historical template.

    Clark, Gangopadhyay, Hay, Rajagopalan and Wilby (2004b), DOI
    10.1175/1525-7541(2004)005<0243:TSSAMF>2.0.CO;2. This implements the
    permutation core; it does not reproduce the publication's forecast models
    or station experiments. See the module References for the complete source.

    Every non-ensemble coordinate (day, variable, site) is treated as one column.
    Columns keep their exact empirical marginals. To impose temporal dependence,
    template rows must be coherent historical trajectories across all days; do
    not draw a different donor set independently for every day.

    ``ties='stable'`` keeps donor order among tied template values;
    ``ties='random'`` breaks them at random (Clark et al. 2004b). With many dry
    template days, stable ordering always hands the smallest wet values to the
    last tied members, a member-dependent bias; random ties avoid it. Exact
    Spearman agreement is not promised where ties exist. All-NaN ensemble columns stay NaN.
    Partial missing columns are rejected because unequal sample sizes have no
    unique rank permutation. Template must be finite for active columns.
    Accepts NumPy arrays, or returns a DataArray when ensemble is a DataArray.
    """
    values = np.asarray(ensemble, dtype=float)
    reference = np.asarray(template, dtype=float)
    if values.shape != reference.shape or values.ndim == 0:
        raise ValueError("ensemble and template must have identical non-scalar shapes")
    axis = int(axis)
    moved = np.moveaxis(values, axis, 0)
    historical = np.moveaxis(reference, axis, 0)
    n = moved.shape[0]
    if n == 0:
        raise ValueError("the ensemble axis cannot be empty")
    flat = moved.reshape(n, -1)
    hist = historical.reshape(n, -1)
    active = ~np.isnan(flat).all(axis=0)
    if not np.isfinite(flat[:, active]).all():
        raise ValueError("active ensemble columns must contain only finite values")
    if not np.isfinite(hist[:, active]).all():
        raise ValueError("template columns corresponding to active ensemble columns must be finite")
    result = np.full_like(flat, np.nan)
    # argsort(template) supplies member slots from lowest to highest rank.
    if ties == "random":
        generator = rng if rng is not None else np.random.default_rng()
        key = generator.random(hist[:, active].shape)
        order = np.lexsort((key, hist[:, active]), axis=0)
    elif ties == "stable":
        order = np.argsort(hist[:, active], axis=0, kind="stable")
    else:
        raise ValueError("ties must be 'stable' or 'random'")
    sorted_values = np.sort(flat[:, active], axis=0)
    shuffled = np.empty_like(sorted_values)
    np.put_along_axis(shuffled, order, sorted_values, axis=0)
    result[:, active] = shuffled
    result = np.moveaxis(result.reshape(moved.shape), 0, axis)
    if isinstance(ensemble, xr.DataArray):
        output = ensemble.copy(data=result)
        output.attrs["rank_reordering"] = f"Schaake shuffle; {ties} tie order"
        return output
    return result


def enso_rank_weights(index, target: float, strength: float = 1.0,
                      selection: float = 1.0) -> np.ndarray:
    """Selection weights from ranked absolute distance to a scalar climate index.

    Clark, Gangopadhyay, Brandon, Werner, Hay, Rajagopalan and Yates (2004a),
    Eq. (9), DOI 10.1029/2003WR002747, building on Yates et al. (2003), DOI
    10.1029/2002WR001769; see full authors/titles in the module References.
    This is the rank-selection component, not the complete Yates kNN generator.

    This explicitly implements the rank-selection rule
    ``rank = floor(N * U**strength / selection) + 1``, with U uniform on [0,1).
    Probabilities are integrated analytically, without Monte Carlo error.
    strength > 1 favors close years; selection >= 1 truncates distant ranks.
    strength=selection=1 is uniform. Missing index values receive zero mass.
    Equal distances share the average mass of their occupied ranks.

    This helper does not estimate skill or tune either parameter. Optimize only
    with independent hindcasts, and do not treat one operational forecast as a
    skill assessment. Index values and target must use the same anomaly baseline.
    """
    values = np.asarray(index, dtype=float)
    if values.ndim != 1 or not np.isfinite(target):
        raise ValueError("index must be one dimensional and target finite")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be finite and positive")
    if not np.isfinite(selection) or selection < 1:
        raise ValueError("selection must be finite and >= 1")
    good = np.flatnonzero(np.isfinite(values))
    if len(good) == 0:
        raise ValueError("at least one finite historical index is required")
    distance = np.abs(values[good] - target)
    order = np.argsort(distance, kind="stable")
    n = len(order)
    edges = np.minimum(selection * np.arange(n + 1, dtype=float) / n, 1)
    mass = np.diff(edges ** (1.0 / strength))
    ranked_distance = distance[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ranked_distance) != 0) + 1]
    stops = np.r_[starts[1:], n]
    for start, stop in zip(starts, stops):
        mass[start:stop] = mass[start:stop].mean()
    result = np.zeros(len(values), dtype=float)
    result[good[order]] = mass
    return result / result.sum()


class ForecastAnalogGenerator:
    """Sample complete observed seasons using shared regional donor weights.

    This is a package baseline based on the Briggs--Wilks historical weighting
    identity (1996), not the daily kNN algorithm of Apipattanavis et al. (2007)
    or the daily marginal resampling/Schaake algorithm of Clark et al. (2004a).

    Local year weights p(category)/N(category) are averaged over forecast-valid
    sites, using cos(latitude) area weights by default. The resulting probability
    distribution is shared by the full field. ``diagnostics_`` quantifies its
    *expected* gridded tercile mismatch, separate from finite-member sampling.
    Daily input NaNs are retained, never converted to dry days. Donor years can
    be incomplete at individual cells; expected valid-season mass is reported.
    """

    def __init__(self, months=(7, 8, 9), climatology=(1991, 2020),
                 tercile_method="empirical", seed=42, area_weighted=True):
        self.months = tuple(months)
        self.climatology = tuple(climatology)
        self.tercile_method = tercile_method
        self.seed = int(seed)
        self.area_weighted = bool(area_weighted)

    def fit(self, observations: xr.Dataset, probabilities: xr.DataArray,
            year_prior: xr.DataArray | None = None, constraints=None, constraint_method="mre",
            constraint_mode="soft", _regional_constraints=True):
        """Fit regional analog weights; optional prior has dimension season_year.

        ``constraints`` (list of SeasonalConstraint) replaces the p/N local
        weights by minimum-relative-entropy (or Croley-inspired) weights that
        target the supplied constraints in the selected exact or soft mode;
        the seasonal total with ``probabilities`` is added unless
        a constraint already uses the main forecast (``probabilities=None``).

        With ``constraints``, the prior is the reference distribution *inside*
        the constrained optimization. Without constraints it multiplies the
        forecast-conditioned donor weights. ``constraint_mode='exact'`` enforces
        feasible constraint probabilities; the default ``'soft'`` retains the
        original tolerance-penalized objective. Diagnostics describe final
        weights actually used for sampling. A prior can be built by
        enso_rank_weights.
        """
        if "PRCP" not in observations:
            raise ValueError("observations must include PRCP")
        self.cube_ = seasonal_cube(observations, months=self.months)
        self.probabilities_ = prepare_probabilities(probabilities, target=observations)
        years = np.asarray(self.cube_.season_year.values)
        if year_prior is not None:
            if not isinstance(year_prior, xr.DataArray) or year_prior.dims != ("season_year",):
                raise ValueError("year_prior must be a DataArray with dimension season_year")
            prior = np.asarray(year_prior.sel(season_year=years).values, dtype=float)
            if not np.isfinite(prior).all() or np.any(prior < 0):
                raise ValueError("year_prior must be finite and nonnegative")
        else:
            prior = None
        totals = self.cube_.PRCP.sum("day", skipna=False).transpose("season_year", "Y", "X")
        ny, nx = totals.sizes["Y"], totals.sizes["X"]
        values = np.asarray(totals.values).reshape(len(years), -1)
        categories, thresholds = classify_seasons(
            values, years, climatology=self.climatology, method=self.tercile_method)
        p = self.probabilities_.transpose("probability", "Y", "X").values.reshape(3, -1)
        local_weights, flags = year_weights(categories, p)
        if constraints:
            # One coherent donor field optimizes regional constraints. Requiring
            # every local target to be feasible first would reject a feasible
            # regional problem. The preliminary local weights are diagnostic.
            local_mode = ("soft" if _regional_constraints and constraint_mode == "exact"
                          else constraint_mode)
            local_weights, flags = self._constrained(observations, years, values, p, constraints,
                                                     constraint_method, (ny, nx), prior,
                                                     local_mode)
        usable = np.isfinite(p).all(axis=0) & (local_weights.sum(axis=0) > 0)
        if not usable.any():
            raise ValueError("No forecast-valid grid cells have classified historical seasons")
        if self.area_weighted:
            latitude = np.asarray(totals.Y.values, dtype=float)
            site_area = np.broadcast_to(np.cos(np.deg2rad(latitude))[:, None], (ny, nx)).ravel()
            site_area = np.clip(site_area, 0, None)
        else:
            site_area = np.ones(ny * nx)
        site_area[~usable] = 0
        if site_area.sum() <= 0:
            raise ValueError("No positive-area forecast-valid sites")
        weights = np.sum(local_weights * site_area[None, :], axis=1) / site_area.sum()
        if constraints and _regional_constraints:
            weights = self._regional_constrained(site_area, prior, constraint_mode)
        if prior is not None and not constraints:
            weights *= prior
        if weights.sum() <= 0:
            raise ValueError("The combined donor weights have no positive donor mass")
        weights = weights / weights.sum()
        coords = {"Y": totals.Y, "X": totals.X}
        self.weights_ = xr.DataArray(weights, dims="season_year", coords={"season_year": years}, name="donor_weight")
        self.local_weights_ = xr.DataArray(local_weights.reshape(len(years), ny, nx),
            dims=("season_year", "Y", "X"), coords={"season_year": years, **coords}, name="local_donor_weight")
        self.thresholds_ = xr.DataArray(thresholds.reshape(2, ny, nx),
            dims=("threshold", "Y", "X"), coords={"threshold": ["q33", "q67"], **coords}, name="thresholds")
        valid_mass = np.sum(weights[:, None] * (categories >= 0), axis=0)
        expected = np.stack([np.sum(weights[:, None] * (categories == c), axis=0)
                             for c in range(3)])
        expected = np.divide(expected, valid_mass[None, :],
                             out=np.full_like(expected, np.nan), where=valid_mass[None, :] > 0)
        expected[:, ~usable] = np.nan
        expected_da = xr.DataArray(expected.reshape(3, ny, nx), dims=("probability", "Y", "X"),
                                  coords={"probability": ["PB", "PN", "PA"], **coords})
        self.valid_mask_ = xr.DataArray(usable.reshape(ny, nx), dims=("Y", "X"), coords=coords)
        self.diagnostics_ = xr.Dataset({
            "target_probability": self.probabilities_,
            "expected_probability": expected_da,
            "expected_probability_error": expected_da - self.probabilities_,
            "expected_valid_season_fraction": xr.DataArray(valid_mass.reshape(ny, nx), dims=("Y", "X"), coords=coords),
            "conditioning_flag": xr.DataArray(np.asarray(flags).reshape(ny, nx), dims=("Y", "X"), coords=coords),
            "thresholds": self.thresholds_,
        }, attrs={"method": "area-average of local donor probabilities; shared whole-season donor",
                  "limitation": "Heterogeneous gridded tercile probabilities generally cannot all be reproduced exactly.",
                  "effective_donor_years": float(1 / np.sum(weights ** 2))})
        if constraints and _regional_constraints:
            info = self._cinfo[0]
            shared = np.stack([
                np.stack([(weights[:, None] * (info["categories"][name] == k)).sum(axis=0)
                          for k in range(3)]).reshape(3, ny, nx)
                for name in info["names"]])
            self.constraint_info_["shared_donor_achieved_probability"] = (
                ("constraint", "probability", "Y", "X"), shared)
            self.constraint_info_["per_site_optimized_probability"] = (
                self.constraint_info_["achieved_probability"].copy())
            self.constraint_info_["per_site_effective_years"] = (
                self.constraint_info_["effective_years"].copy())
            self.constraint_info_["achieved_probability"] = (
                ("constraint", "probability", "Y", "X"), shared)
            self.constraint_info_["effective_years"] = (
                ("Y", "X"), np.full((ny, nx), 1 / np.sum(weights ** 2)))
            self.constraint_info_.attrs["achieved_probability_definition"] = (
                "achieved_probability and shared_donor_achieved_probability describe "
                "the shared donors actually sampled; per_site_optimized_probability "
                "describes the preliminary local optimization")
            self.constraint_info_.attrs["regional_constraint_mode"] = constraint_mode
        return self

    def _constrained(self, observations, years, totals, p, constraints, method, shape,
                     prior, constraint_mode):
        from .mre import constrained_year_weights, relabel_probabilities, thresholds_for
        names = [con.name for con in constraints]
        if len(set(names)) != len(names):
            raise ValueError("constraint names must be unique")
        if sum(con.probabilities is None for con in constraints) > 1:
            raise ValueError("Only one constraint may use the main forecast probabilities")
        values, probs, tols, primary, thr_fixed = {}, {}, {}, None, {}
        for con in constraints:
            if con.probabilities is None:
                q, primary = p, con.name
            else:
                q = prepare_probabilities(relabel_probabilities(con.probabilities), target=observations)
                q = q.transpose("probability", "Y", "X").values.reshape(3, -1)
            v = con.attribute.compute(observations["PRCP"], years)
            values[con.name] = v.transpose("season_year", "Y", "X").values.reshape(len(years), -1)
            probs[con.name], tols[con.name] = q, con.tolerance
            fixed = thresholds_for(con, target=observations, n_sites=q.shape[1])
            if fixed is not None:
                thr_fixed[con.name] = fixed
        if primary is None:
            primary = "total" if "total" not in values else "forecast_total"
            values[primary], probs[primary], tols[primary] = totals, p, 0.01
        weights, flags, info = constrained_year_weights(values, probs, years, climatology=self.climatology,
                                                        prior=prior, tolerances=tols, method=method,
                                                        constraint_mode=constraint_mode,
                                                        tercile_method=self.tercile_method, thresholds=thr_fixed)
        self._cinfo = (info, probs, tols, method)
        coords = {"constraint": info["names"], "probability": ["PB", "PN", "PA"],
                  "Y": self.probabilities_.Y, "X": self.probabilities_.X}
        self.constraint_info_ = xr.Dataset({
            "target_probability": (("constraint", "probability", "Y", "X"),
                                   np.stack([info["target_probability"][k].reshape(3, *shape) for k in info["names"]])),
            "achieved_probability": (("constraint", "probability", "Y", "X"),
                                     np.stack([info["achieved_probability"][k].reshape(3, *shape) for k in info["names"]])),
            "effective_years": (("Y", "X"), info["effective_years"].reshape(shape))},
            coords=coords, attrs={"method": method, "primary_constraint": primary,
                                 "constraint_mode": constraint_mode})
        return weights, flags

    def _regional_constrained(self, site_area, prior, constraint_mode):
        """One shared donor distribution fitted to AREA-MEAN constraints.

        Features are area-weighted fractions of cells in each class for every
        year; targets are area-weighted forecast probabilities. Averaging the
        per-cell weights instead would not honour any constraint in general.
        """
        from .mre import solve_weights
        info, probs, tols, method = self._cinfo
        a = site_area / site_area.sum()
        feats, targets, taus = [], [], []
        valid = np.ones(len(self.cube_.season_year), dtype=bool)
        for name in info["names"]:
            c = info["categories"][name]
            valid &= (c[:, a > 0] >= 0).all(axis=1)
            for k in range(3):
                feats.append(((c == k) * a).sum(1))
                targets.append(float(np.nansum(probs[name][k] * a)))
                taus.append(tols[name])
        if not valid.any():
            raise ValueError("No donor season has complete constrained attributes across the forecast-valid region")
        g = np.stack(feats, -1)[:, None, :]
        regional_prior = valid.astype(float) * (prior if prior is not None else 1.)
        if regional_prior.sum() <= 0:
            raise ValueError("No complete regional donor has positive year_prior support")
        w, rinfo = solve_weights(g, np.asarray(targets)[None], regional_prior[:, None],
                                 np.asarray(taus), method=method,
                                 constraint_mode=constraint_mode)
        self.constraint_info_.attrs["regional_effective_years"] = float(rinfo["effective_years"][0])
        self.constraint_info_.attrs["regional_achieved_minus_target_max"] = float(np.max(np.abs(rinfo["achieved"][0] - np.asarray(targets))))
        return w[:, 0]

    def generate(self, year: int, n_members: int = 20) -> xr.Dataset:
        """Return observed donor trajectories as member,T,Y,X, without perturbation."""
        if not hasattr(self, "weights_"):
            raise RuntimeError("Call fit before generate")
        if not isinstance(n_members, (int, np.integer)) or n_members < 1:
            raise ValueError("n_members must be a positive integer")
        dates = season_dates(year, self.months)
        if len(dates) != self.cube_.sizes["day"]:
            raise ValueError("Target dates and historical season length differ")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(year)]))
        index = rng.choice(len(self.weights_), size=n_members, p=self.weights_.values)
        out = self.cube_.isel(season_year=xr.DataArray(index, dims="member"))
        out = out.rename({"day": "T", "season_year": "donor_year"})
        out = out.assign_coords(T=dates, member=np.arange(n_members))
        out = out.where(self.valid_mask_).transpose("member", "T", "Y", "X", missing_dims="ignore")
        out.attrs.update({"generator": "shared whole-season forecast analog",
                          "season_start_year": int(year), "seed": self.seed,
                          "climatology_start": int(self.climatology[0]),
                          "climatology_end": int(self.climatology[1]),
                          "conditioning_limitation": "Regional donor weights approximate heterogeneous local forecasts; inspect diagnostics_.",
                          "calendar_policy": "Gregorian; February 29 omitted"})
        return out


class ForecastSchaakeGenerator(ForecastAnalogGenerator):
    """Clark et al. (2004a)-inspired resampling with coherent Schaake ranks.

    DOI 10.1029/2003WR002747; complete authors and title are in the module
    References. The daily marginal-resampling and rank-permutation cores are
    implemented; the template convention below is a calendar-aligned variant.
    The authors' station experiments and RPSS parameter optimization are not
    reproduced by this class.

    At each cell/day/member, a year is drawn from that cell's forecast-conditioned
    year weights (p/N, or Stedinger–Kim pdf-ratio, optionally multiplied by a
    ``year_prior`` such as :func:`enso_rank_weights`, implementing Clark's
    rank-selection component), then a day uniformly within +/- ``window`` seasonal positions
    (truncated at season boundaries). Every variable's marginal ensemble is then
    reordered to common historical-year templates that advance together through
    the season, which restores spatial, temporal and inter-variable rank
    dependence from those templates, subject to ties. These templates use the
    same seasonal day as the target; unlike Clark's original date selection,
    their initial dates do not have random +/- ``window`` offsets.
    After reordering, TMIN<=TMAX and HUMIN<=HUMAX are enforced by
    swapping (counts in ``attrs``). Those physical swaps may change the sorted
    marginals after shuffling, so inspect the reported counts when validating.

    Templates are distinct years when ``n_members`` does not exceed the number of
    complete template years; otherwise they are drawn with replacement (reported).
    Seasonal tercile frequencies are not guaranteed by weighted daily sampling;
    evaluate them with ``ensemble_diagnostics``. Seasonal data are loaded eagerly.
    """

    def __init__(self, months=(7, 8, 9), climatology=(1991, 2020),
                 tercile_method="empirical", seed=42, window=7, weighting="tercile", ties="random",
                 template_years="uniform", pdf_ratio_calibrate=True):
        super().__init__(months, climatology, tercile_method, seed)
        if not isinstance(window, (int, np.integer)) or window < 0:
            raise ValueError("window must be a nonnegative integer")
        if weighting not in {"tercile", "pdf_ratio"}:
            raise ValueError("weighting must be 'tercile' or 'pdf_ratio'")
        if ties not in {"stable", "random"}:
            raise ValueError("ties must be 'stable' or 'random'")
        if template_years not in {"uniform", "weighted"}:
            raise ValueError("template_years must be 'uniform' or 'weighted'")
        self.window, self.weighting, self.ties = int(window), weighting, ties
        self.template_years = template_years
        self.pdf_ratio_calibrate = bool(pdf_ratio_calibrate)

    def fit(self, observations: xr.Dataset, probabilities: xr.DataArray,
            year_prior: xr.DataArray | None = None, constraints=None, constraint_method="mre",
            constraint_mode="soft"):
        """``constraints``: see :meth:`ForecastAnalogGenerator.fit`. The day-to-day
        sequencing (onset, dry spells) comes from the Schaake templates, so use
        ``template_years='weighted'`` when such constraints matter."""
        super().fit(observations, probabilities, year_prior=year_prior,
                    constraints=constraints, constraint_method=constraint_method,
                    constraint_mode=constraint_mode, _regional_constraints=False)
        self.variables_ = ["PRCP"] + [v for v in self.cube_.data_vars if v != "PRCP"]
        years = np.asarray(self.cube_.season_year.values)
        nyears = len(years)
        shape = (nyears, self.cube_.sizes["day"], -1)
        self._values = {v: self.cube_[v].transpose("season_year", "day", "Y", "X").values.reshape(shape)
                        for v in self.variables_}
        local = self.local_weights_.values.reshape(nyears, -1).copy()
        if self.weighting == "pdf_ratio" and not constraints:
            rain = self._values["PRCP"]
            totals = np.where(np.isfinite(rain).all(axis=1), np.nansum(rain, axis=1), np.nan)
            cats, _ = classify_seasons(totals, years, climatology=self.climatology, method=self.tercile_method)
            z = normal_scores(totals, years, climatology=self.climatology, method=self.tercile_method)
            p = self.probabilities_.transpose("probability", "Y", "X").values.reshape(3, -1)
            local, flags = tercile_pdf_ratio_weights(z, cats, p,
                                                       calibrate=self.pdf_ratio_calibrate)
            self.diagnostics_["conditioning_flag"] = (("Y", "X"),
                flags.reshape(self.cube_.sizes["Y"], self.cube_.sizes["X"]))
        if year_prior is not None and not constraints:
            local = local * year_prior.sel(season_year=years).values[:, None]
        total = local.sum(axis=0)
        self._local = np.divide(local, total, out=np.zeros_like(local), where=total > 0)
        active = self.valid_mask_.values.ravel() & (total > 0)
        if not active.any():
            raise ValueError("The prior leaves no forecast-valid site with positive donor mass")
        self._active = active
        ny, nx = self.cube_.sizes["Y"], self.cube_.sizes["X"]
        self.valid_mask_ = self.valid_mask_.copy(data=active.reshape(ny, nx))
        self.local_weights_ = self.local_weights_.copy(data=self._local.reshape(nyears, ny, nx))
        rain_totals = self.cube_.PRCP.sum("day", skipna=False).values.reshape(nyears, -1)
        donor_categories, _ = classify_seasons(
            rain_totals, years, climatology=self.climatology, method=self.tercile_method)
        donor_probability = np.stack([
            np.sum(self._local * (donor_categories == c), axis=0) for c in range(3)])
        donor_probability[:, ~active] = np.nan
        self.diagnostics_["daily_donor_category_probability"] = (
            ("probability", "Y", "X"), donor_probability.reshape(3, ny, nx))
        self.diagnostics_["daily_donor_category_probability_error"] = (
            self.diagnostics_["daily_donor_category_probability"] - self.probabilities_)
        self.diagnostics_["daily_donor_effective_years"] = (
            ("Y", "X"), np.divide(1., np.sum(self._local ** 2, axis=0),
                                    out=np.full(ny * nx, np.nan), where=active).reshape(ny, nx))
        latitude = np.asarray(self.cube_.Y.values, dtype=float)
        site_area = (np.broadcast_to(np.cos(np.deg2rad(latitude))[:, None], (ny, nx)).ravel()
                     if self.area_weighted else np.ones(ny * nx))
        site_area = np.clip(site_area, 0., None)
        site_area[~active] = 0.
        if site_area.sum() <= 0:
            raise ValueError("No positive-area forecast-valid sites remain after applying the prior")
        donor_weights = (self._local * site_area[None, :]).sum(axis=1) / site_area.sum()
        self.weights_ = self.weights_.copy(data=donor_weights)
        self.diagnostics_.attrs["effective_donor_years"] = float(1 / np.sum(donor_weights ** 2))
        if constraints:
            self.constraint_info_.attrs.pop("regional_effective_years", None)
            self.constraint_info_.attrs.pop("regional_achieved_minus_target_max", None)
            self.constraint_info_.attrs["achieved_probability_definition"] = (
                "per-site donor weights used for daily marginal resampling; "
                "generated seasonal attributes may differ after the Schaake shuffle")
        complete = np.ones(nyears, dtype=bool)
        for v in self.variables_:
            complete &= np.isfinite(self._values[v][:, :, active]).all(axis=(1, 2))
        self._template_indices = np.flatnonzero(complete)
        if len(self._template_indices) < 2:
            raise ValueError("Schaake templates require at least two seasons complete across all active cells; reduce the domain or resolve missing observations.")
        if self.template_years == "weighted" and self._local[self._template_indices][:, active].sum() <= 0:
            raise ValueError("No complete Schaake template has positive forecast/prior weight; use uniform templates or revise the prior")
        self.diagnostics_ = self.diagnostics_.drop_vars([
            "expected_probability", "expected_probability_error", "expected_valid_season_fraction"])
        self.diagnostics_.attrs.update({
            "method": "Clark et al. (2004a): local forecast-weighted daily resampling + advancing common-year Schaake templates",
            "template_year_count": int(len(self._template_indices)), "weighting": self.weighting,
            "pdf_ratio_calibrate": self.pdf_ratio_calibrate,
            "year_prior": "applied" if year_prior is not None else "none",
            "limitation": "Seasonal tercile probabilities are approximate; assess generated seasonal totals.",
            "window_days": self.window})
        return self

    def generate(self, year: int, n_members: int = 20) -> xr.Dataset:
        if not hasattr(self, "_template_indices"):
            raise RuntimeError("Call fit before generate")
        if not isinstance(n_members, (int, np.integer)) or n_members < 1:
            raise ValueError("n_members must be a positive integer")
        dates = season_dates(year, self.months)
        nyears, ndays, nsite = self._values["PRCP"].shape
        ny, nx = self.cube_.sizes["Y"], self.cube_.sizes["X"]
        if len(dates) != ndays:
            raise ValueError("Target dates and historical season length differ")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(year), 311]))
        distinct = n_members <= len(self._template_indices)
        if self.template_years == "weighted":
            # Template years drawn with the area-mean forecast weights: carries the
            # conditioned day-to-day sequencing (onset, dry spells) into the ranks.
            area = np.cos(np.deg2rad(np.broadcast_to(self.cube_.Y.values[:, None], (ny, nx)).ravel()))
            tw = (self._local[:, self._active] * area[self._active]).sum(axis=1)[self._template_indices]
            tw = tw / tw.sum()
            distinct = distinct and np.count_nonzero(tw) >= n_members
            donor_indices = rng.choice(self._template_indices, size=n_members, replace=not distinct, p=tw)
        else:
            donor_indices = rng.choice(self._template_indices, size=n_members, replace=not distinct)
        day = np.arange(ndays)
        lo, hi = np.maximum(0, day - self.window), np.minimum(ndays, day + self.window + 1)
        marginal = {v: np.full((n_members, ndays, nsite), np.nan) for v in self.variables_}
        unresolved = 0
        for site in np.flatnonzero(self._active):
            cumulative = np.cumsum(self._local[:, site])
            cumulative /= cumulative[-1]
            def donors(n):
                y = np.minimum(np.searchsorted(cumulative, rng.random(n), side="right"), nyears - 1)
                return y
            y = donors(n_members * ndays).reshape(n_members, ndays)
            d = (lo + np.floor(rng.random((n_members, ndays)) * (hi - lo)).astype(int))
            for v in self.variables_:
                marginal[v][:, :, site] = self._values[v][y, d, site]
            # Sample the exact conditional donor distribution for missing draws.
            # Fixed retry limits can discard valid sites merely because complete
            # donor days are rare; shared (year, day) draws retain variable links.
            bad = np.zeros((n_members, ndays), dtype=bool)
            for v in self.variables_:
                bad |= ~np.isfinite(marginal[v][:, :, site])
            joint = np.logical_and.reduce([np.isfinite(self._values[v][:, :, site]) for v in self.variables_])
            failed = False
            for position in np.flatnonzero(bad.any(axis=0)):
                donor_mass = joint[:, lo[position]:hi[position]] * self._local[:, site, None]
                mass = donor_mass.sum()
                if mass <= 0:
                    failed = True
                    break
                members = np.flatnonzero(bad[:, position])
                draw = rng.choice(donor_mass.size, size=len(members), p=donor_mass.ravel() / mass)
                yb, db = np.unravel_index(draw, donor_mass.shape)
                db += lo[position]
                for v in self.variables_:
                    marginal[v][members, position, site] = self._values[v][yb, db, site]
            if failed:
                unresolved += 1
                for v in self.variables_:
                    marginal[v][:, :, site] = np.nan
        out, swaps = {}, {}
        for v in self.variables_:
            templates = self._values[v][donor_indices]
            out[v] = schaake_shuffle(marginal[v], templates, ties=self.ties, rng=rng)
        for low, high in (("TMIN", "TMAX"), ("HUMIN", "HUMAX")):
            if low in out and high in out:
                crossed = np.isfinite(out[low]) & np.isfinite(out[high]) & (out[low] > out[high])
                swaps[f"{low}_{high}_swaps"] = int(crossed.sum())
                out[low], out[high] = np.where(crossed, out[high], out[low]), np.where(crossed, out[low], out[high])
        result = xr.Dataset({v: (("member", "T", "Y", "X"), out[v].reshape(n_members, ndays, ny, nx))
                             for v in self.variables_},
                            coords={"member": np.arange(n_members), "T": dates,
                                    "Y": self.cube_.Y, "X": self.cube_.X,
                                    "template_year": ("member", self.cube_.season_year.values[donor_indices])})
        for v in self.variables_:
            result[v].attrs = dict(self.cube_[v].attrs)
        result.attrs.update({"generator": "Clark et al. (2004a) forecast-weighted resampling with Schaake shuffle",
                             "season_start_year": int(year), "seed": self.seed,
                             "window_days": self.window, "weighting": self.weighting, "tie_order": self.ties,
                             "conditioning_limitation": "Weighted daily marginal sampling does not enforce exact seasonal tercile probabilities.",
                             "template_weighting": self.template_years,
                             "template_sampling": ("distinct complete historical years" if distinct else
                                                   "complete historical years WITH replacement (n_members > template years)"),
                             "cells_masked_unresolved_missing": unresolved,
                             **{k: int(v) for k, v in swaps.items()},
                             "calendar_policy": "Gregorian; February 29 omitted"})
        return result
