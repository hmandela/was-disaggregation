"""Apipattanavis et al. (2007): three-state Markov plus daily multisite kNN.

Primary source: https://doi.org/10.1029/2006WR005714, sections 2.1--2.4.
Historical predecessor/successor pairs are filtered on *both* precipitation
states. A sampled successor is copied jointly across variables and sites.
Numerical extensions (dimensionless distance, sparse-window expansion and wet
conditional extreme thresholds) are explicit options, never hidden changes.

References
----------
* Somkiat Apipattanavis, Guillermo Podesta, Balaji Rajagopalan and Richard W.
  Katz (2007), "A Semiparametric Multivariate and Multisite Weather Generator".
  https://doi.org/10.1029/2006WR005714
  Core: monthly three-state Markov occurrence, predecessor/successor state
  filtering, daily nearest-neighbor selection and joint successor copying;
  conditional bootstrap as in sections 2.1--2.4. Standardized distance,
  support/window expansion, wet-conditional extreme thresholds and integer
  largest-remainder allocation are declared package variants/extensions.
* Balaji Rajagopalan and Upmanu Lall (1999), "A k-Nearest-Neighbor Simulator
  for Daily Precipitation and Other Weather Variables".
  https://doi.org/10.1029/1999WR900028
  Background: daily successor resampling; this module does not provide a
  separate reproduction of that article's original generator.
* William M. Briggs and Daniel S. Wilks (1996), "Extension of the Climate
  Prediction Center Long-Lead Temperature and Precipitation Outlooks to General
  Weather Statistics". https://doi.org/10.1175/1520-0442(1996)009<3496:EOTCPC>2.0.CO;2
  Core component: forecast-conditioned historical bootstrap category masses.

The code implements a specified regional algorithm, not the original station
experiments. Published method authors are distinct from software authorship
recorded in package metadata and copyright notices.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from .conditioning import classify_seasons, year_weights
from .data import seasonal_cube, season_dates, validate_months


def _states(rain, threshold, extreme):
    return np.where(rain < threshold, 0, np.where(rain > extreme, 2, 1)).astype(np.int8)


class ApipattanavisGenerator:
    """A lag-1 semiparametric generator for a homogeneous station/grid region.

    Somkiat Apipattanavis, Guillermo Podesta, Balaji Rajagopalan and Richard W.
    Katz (2007), DOI 10.1029/2006WR005714; see the module References for the
    title and related sources. Markov/kNN and joint-successor cores follow the
    article; numerical choices below are declared variants. Reproduction of
    the authors' data, station experiments and forecast skill is not claimed.

    ``window=3`` means a seven-day predecessor window, as in the paper.
    ``k=None`` uses floor(sqrt(N)) for N eligible pairs. Rank probabilities are
    (1/rank)/sum(1/rank). ``standardize=False`` and equal variable weights give
    the paper's direct Euclidean distance; True gives a dimensionless extension.
    ``extreme_basis='wet'`` interprets the 80th percentile conditionally on wet
    amounts (consistent with the dry=.85/wet=.12/extreme=.03 paper example).
    ``'all'`` follows the literal all-daily-amount wording; a very dry month may
    then have no ordinary wet state. Thresholds are fitted to the actual sample.

    A regional forecast triplet PB/PN/PA bootstraps 100 historical seasons by
    default, then refits both components as in section 2.4. Largest remainders
    allocate integer class counts. Forecast-conditioned donor counts do not
    guarantee the generated seasonal totals' category probabilities.

    Complete joint observations are required on active cells; all-missing cells
    remain masked. This avoids turning missing measurements into dry days.
    ``empty_candidates='expand'`` first expands an empty window to the available
    season and removes transitions without any state-consistent historical pair
    (reported in attrs). ``'raise'`` rejects an empty local pool. No state is
    silently replaced by another one. Spatial heterogeneity and extremes beyond
    observed joint fields remain limitations of the original resampling model.
    """

    def __init__(self, months=(7, 8, 9), climatology=(1991, 2020),
                 wet_threshold=.3, extreme_quantile=.8, extreme_basis="wet",
                 window=3, k=None, standardize=True, variable_weights=None,
                 site_weights=None, bootstrap_years=100,
                 empty_candidates="expand", seed=42):
        self.months = validate_months(months)
        self.climatology = tuple(climatology)
        if not np.isfinite(wet_threshold) or wet_threshold <= 0:
            raise ValueError("wet_threshold must be finite and positive")
        if not np.isfinite(extreme_quantile) or not 0 < extreme_quantile < 1:
            raise ValueError("extreme_quantile must lie strictly between zero and one")
        if extreme_basis not in {"wet", "all"}:
            raise ValueError("extreme_basis must be 'wet' or 'all'")
        if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window < 0:
            raise ValueError("window must be a nonnegative integer")
        if k is not None and (isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1):
            raise ValueError("k must be a positive integer or None")
        if (isinstance(bootstrap_years, (bool, np.bool_)) or
                not isinstance(bootstrap_years, (int, np.integer)) or bootstrap_years < 3):
            raise ValueError("bootstrap_years must be an integer >= 3")
        if empty_candidates not in {"expand", "raise"}:
            raise ValueError("empty_candidates must be 'expand' or 'raise'")
        self.wet_threshold, self.extreme_quantile = float(wet_threshold), float(extreme_quantile)
        self.extreme_basis, self.window, self.k = extreme_basis, int(window), k
        self.standardize = bool(standardize)
        self.variable_weights, self.site_weights = variable_weights, site_weights
        self.bootstrap_years, self.empty_candidates, self.seed = int(bootstrap_years), empty_candidates, int(seed)

    def fit(self, observations: xr.Dataset, probabilities=None, year_prior=None):
        """Fit to T,Y,X observations; forecast is one regional PB,PN,PA triplet.

        ``year_prior`` optionally supplies (season_year,) nonnegative within-class
        donor preferences. A forecast is mandatory if this prior is used. Spatial
        forecasts are rejected instead of being treated as a single regional
        target. Units must already be canonical (rain in mm/day).
        """
        if not isinstance(observations, xr.Dataset) or "PRCP" not in observations:
            raise ValueError("observations must be a Dataset containing PRCP")
        self.cube_ = seasonal_cube(observations, self.months)
        self.variables_ = ["PRCP"] + [v for v in self.cube_.data_vars if v != "PRCP"]
        self.years_ = np.asarray(self.cube_.season_year.values)
        self._values = np.stack([self.cube_[v].values for v in self.variables_], axis=-1)
        ny, nd, nlat, nlon, nv = self._values.shape
        flat = self._values.reshape(ny, nd, nlat * nlon, nv)
        all_missing = np.isnan(flat).all(axis=(0, 1, 3))
        complete = np.isfinite(flat).all(axis=(0, 1, 3))
        if np.any(~all_missing & ~complete):
            raise ValueError("Active cells need complete joint daily observations; resolve partial missing/nonfinite data explicitly")
        if not complete.any():
            raise ValueError("No complete joint cell is available")
        if np.any(flat[:, :, complete, 0] < 0):
            raise ValueError("PRCP must be nonnegative")
        for low, high in (("TMIN", "TMAX"), ("HUMIN", "HUMAX")):
            if low in self.variables_ and high in self.variables_:
                if np.any(flat[:, :, complete, self.variables_.index(low)] >
                          flat[:, :, complete, self.variables_.index(high)]):
                    raise ValueError(f"Observed {low} exceeds {high}; resolve physical inconsistencies before joint resampling")
        if self.site_weights is None:
            area = complete.astype(float)
        else:
            area = np.asarray(self.site_weights, float)
            if area.shape not in {(nlat, nlon), (nlat * nlon,)}:
                raise ValueError("site_weights must match the Y,X grid")
            area = area.reshape(-1)
            if not np.isfinite(area).all() or np.any(area < 0):
                raise ValueError("site_weights must be finite and nonnegative")
            area = area * complete
        if area.sum() <= 0:
            raise ValueError("site_weights have no positive mass on complete cells")
        self._area = area / area.sum()
        self._active = complete
        self._regional = np.einsum("ydsv,s->ydv", np.where(np.isfinite(flat), flat, 0), self._area)
        seasonal_totals = self._regional[:, :, 0].sum(axis=1)
        categories, limits = classify_seasons(seasonal_totals[:, None], self.years_, self.climatology)
        self.thresholds_ = limits[:, 0]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 712]))
        if probabilities is None:
            if year_prior is not None:
                raise ValueError("year_prior requires a regional forecast triplet")
            sample = np.arange(ny)
            target = np.full(3, np.nan)
            counts = np.bincount(categories[categories >= 0], minlength=3)
            donor_weights = np.full(ny, 1 / ny)
        else:
            if isinstance(probabilities, xr.DataArray) and probabilities.dims == ("probability",):
                probabilities = probabilities.sel(probability=["PB", "PN", "PA"]).values
            p = np.asarray(probabilities, float)
            if p.shape != (3,):
                raise ValueError("probabilities must be one regional PB,PN,PA triplet, not a heterogeneous grid")
            local, _ = year_weights(categories, p[:, None], empty_policy="raise")
            if local.sum() <= 0:
                raise ValueError("The regional forecast is invalid or has no classified history")
            target = p / p.sum()
            donor_weights = local[:, 0]
            if year_prior is not None:
                if isinstance(year_prior, xr.DataArray):
                    if year_prior.dims != ("season_year",):
                        raise ValueError("year_prior must have only season_year dimension")
                    prior = np.asarray(year_prior.sel(season_year=self.years_).values, float)
                else:
                    prior = np.asarray(year_prior, float)
                if prior.shape != (ny,) or not np.isfinite(prior).all() or np.any(prior < 0):
                    raise ValueError("year_prior must be finite, nonnegative and match historical seasons")
                for state in range(3):
                    inside = categories[:, 0] == state
                    mass = prior[inside].sum()
                    if target[state] > 0 and mass <= 0:
                        raise ValueError(f"year_prior excludes forecast category {state}")
                    donor_weights[inside] = prior[inside] * target[state] / mass if mass > 0 else 0
            raw_counts = self.bootstrap_years * target
            counts = np.floor(raw_counts).astype(int)
            remainder = self.bootstrap_years - counts.sum()
            counts[np.argsort(-(raw_counts - counts), kind="stable")[:remainder]] += 1
            pieces = []
            for state, number in enumerate(counts):
                if number:
                    inside = np.flatnonzero(categories[:, 0] == state)
                    mass = donor_weights[inside]
                    pieces.append(rng.choice(inside, number, replace=True, p=mass / mass.sum()))
            sample = np.concatenate(pieces)
            rng.shuffle(sample)
        self.fit_sample_indices_ = sample
        self.fit_sample_years_ = self.years_[sample]
        self.donor_weights_ = donor_weights
        self._sample = self._regional[sample]
        self._months = np.asarray(self.cube_.month.values, int)
        self.extreme_thresholds_ = {}
        for month in self.months:
            rain = self._sample[:, self._months == month, 0].ravel()
            if self.extreme_basis == "wet":
                rain = rain[rain >= self.wet_threshold]
            self.extreme_thresholds_[month] = (float(np.quantile(rain, self.extreme_quantile))
                                               if rain.size else self.wet_threshold)
        extreme = np.array([self.extreme_thresholds_[m] for m in self._months])
        self.states_ = _states(self._sample[:, :, 0], self.wet_threshold, extreme)
        self.initial_probabilities_ = np.zeros((12, 3))
        self.transition_probabilities_ = np.zeros((12, 3, 3))
        transition_counts = np.zeros((12, 3, 3), int)
        fallback_rows = np.zeros((12, 3), bool)
        for month in self.months:
            index = month - 1
            frequency = np.bincount(self.states_[:, self._months == month].ravel(), minlength=3)
            self.initial_probabilities_[index] = frequency / frequency.sum()
            pair_day = np.flatnonzero(self._months[1:] == month)
            for origin in range(3):
                previous = self.states_[:, pair_day]
                following = self.states_[:, pair_day + 1]
                row = np.bincount(following[previous == origin], minlength=3)
                transition_counts[index, origin] = row
                if row.sum():
                    self.transition_probabilities_[index, origin] = row / row.sum()
                else:
                    self.transition_probabilities_[index, origin] = frequency / frequency.sum()
                    fallback_rows[index, origin] = True
        scale = self._sample.reshape(-1, nv).std(axis=0, ddof=1) if self.standardize else np.ones(nv)
        self._scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1)
        if self.variable_weights is None:
            self._metric = np.ones(nv)
        else:
            unknown = set(self.variable_weights) - set(self.variables_)
            if unknown:
                raise ValueError(f"Unknown distance variables: {sorted(unknown)}")
            self._metric = np.array([self.variable_weights.get(v, 1) for v in self.variables_], float)
            if not np.isfinite(self._metric).all() or np.any(self._metric < 0) or self._metric.sum() <= 0:
                raise ValueError("variable_weights must be finite, nonnegative and contain positive mass")
        self.diagnostics_ = xr.Dataset({
            "transition_probability": (("month", "origin_state", "next_state"), self.transition_probabilities_),
            "transition_count": (("month", "origin_state", "next_state"), transition_counts),
            "initial_probability": (("month", "next_state"), self.initial_probabilities_),
            "transition_row_fallback": (("month", "origin_state"), fallback_rows),
            "target_probability": ("probability", target),
            "bootstrap_class_count": ("probability", counts),
            "bootstrap_class_probability": ("probability", counts / len(sample)),
            "extreme_threshold": ("month", [self.extreme_thresholds_.get(m, np.nan) for m in range(1, 13)]),
        }, coords={"month": np.arange(1, 13), "origin_state": [0, 1, 2],
                   "next_state": [0, 1, 2], "probability": ["PB", "PN", "PA"]},
            attrs={"method": "Apipattanavis et al. 2007: Markov + lag-1 daily kNN",
                   "source": "https://doi.org/10.1029/2006WR005714",
                   "standardized_distance": int(self.standardize), "extreme_basis": self.extreme_basis,
                   "fit_season_count": len(sample),
                   "limitation": "One representative regional Markov state; generated seasonal probabilities and local spells require validation"})
        return self

    def _pair_pool(self, previous_day, origin, following, expand=False):
        nd = self._sample.shape[1]
        candidates = np.arange(nd - 1)
        if not expand:
            candidates = candidates[np.abs(candidates - previous_day) <= self.window]
        if not len(candidates):
            return np.empty(0, int), np.empty(0, int)
        # At month boundaries, classify amounts with the thresholds of the
        # target days. A neighboring historical day can lie in another month;
        # copying its stored month label would violate the generated state.
        qprev = self.extreme_thresholds_[self._months[previous_day]]
        qnext = self.extreme_thresholds_[self._months[previous_day + 1]]
        first = _states(self._sample[:, candidates, 0], self.wet_threshold, qprev)
        second = _states(self._sample[:, candidates + 1, 0], self.wet_threshold, qnext)
        rows, positions = np.nonzero((first == origin) & (second == following))
        return rows, candidates[positions]

    def generate(self, year: int, n_members=20) -> xr.Dataset:
        """Generate member,T,Y,X fields and record joint donor dates and states."""
        if not hasattr(self, "_sample"):
            raise RuntimeError("Call fit before generate")
        if isinstance(n_members, (bool, np.bool_)) or not isinstance(n_members, (int, np.integer)) or n_members < 1:
            raise ValueError("n_members must be a positive integer")
        dates = season_dates(year, self.months)
        nd = self._sample.shape[1]
        if len(dates) != nd:
            raise ValueError("Generated calendar length differs from historical seasons")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(year), 713]))
        donor_year = np.zeros((n_members, nd), int)
        donor_day = np.zeros((n_members, nd), int)
        state = np.zeros((n_members, nd), np.int8)
        expansions, support_changes = 0, 0
        for member in range(n_members):
            first_state = rng.choice(3, p=self.initial_probabilities_[self._months[0] - 1])
            positions = np.arange(min(nd, self.window + 1))
            initial = _states(self._sample[:, positions, 0], self.wet_threshold,
                              self.extreme_thresholds_[self._months[0]])
            rows, columns = np.nonzero(initial == first_state)
            if not len(rows) and self.empty_candidates == "expand":
                positions = np.arange(nd)
                initial = _states(self._sample[:, :, 0], self.wet_threshold,
                                  self.extreme_thresholds_[self._months[0]])
                rows, columns = np.nonzero(initial == first_state)
                expansions += 1
            if not len(rows):
                raise ValueError("No historical initial weather vector supports the sampled state")
            selected = rng.integers(len(rows))
            sample_row, day = rows[selected], positions[columns[selected]]
            donor_year[member, 0], donor_day[member, 0] = self.fit_sample_indices_[sample_row], day
            state[member, 0] = first_state
            current = self._sample[sample_row, day]
            for step in range(1, nd):
                origin = int(state[member, step - 1])
                probabilities = self.transition_probabilities_[self._months[step] - 1, origin].copy()
                pools = [self._pair_pool(step - 1, origin, j) for j in range(3)]
                used_expansion = [False] * 3
                if self.empty_candidates == "expand":
                    for j in range(3):
                        if probabilities[j] > 0 and not len(pools[j][0]):
                            pools[j] = self._pair_pool(step - 1, origin, j, expand=True)
                            used_expansion[j] = True
                    supported = np.array([len(pool[0]) > 0 for pool in pools])
                    if np.any((probabilities > 0) & ~supported):
                        probabilities *= supported
                        support_changes += 1
                        if probabilities.sum() <= 0:
                            raise ValueError("No state-consistent historical successor supports the Markov transition")
                        probabilities /= probabilities.sum()
                following = int(rng.choice(3, p=probabilities))
                rows, days = pools[following]
                if not len(rows):
                    raise ValueError(f"Empty kNN window for state pair {origin}->{following} at generated day {step}")
                expansions += int(used_expansion[following])
                distance = np.sum(((self._sample[rows, days] - current) /
                                   self._scale * self._metric) ** 2, axis=-1)
                order = np.lexsort((rng.random(len(rows)), distance))
                count = min(len(order), self.k if self.k is not None else max(1, int(np.sqrt(len(order)))))
                kernel = 1 / np.arange(1, count + 1, dtype=float)
                selected = order[rng.choice(count, p=kernel / kernel.sum())]
                sample_row, day = rows[selected], days[selected] + 1
                donor_year[member, step], donor_day[member, step] = self.fit_sample_indices_[sample_row], day
                state[member, step] = following
                current = self._sample[sample_row, day]
        sampled = self._values[donor_year, donor_day]
        out = xr.Dataset({v: (("member", "T", "Y", "X"), sampled[..., j])
                          for j, v in enumerate(self.variables_)},
                         coords={"member": np.arange(n_members), "T": dates,
                                 "Y": self.cube_.Y, "X": self.cube_.X,
                                 "regional_precipitation_state": (("member", "T"), state),
                                 "donor_season_year": (("member", "T"), self.years_[donor_year]),
                                 "donor_day": (("member", "T"), donor_day)})
        for v in self.variables_:
            out[v].attrs = dict(self.cube_[v].attrs)
        out.attrs.update(generator="Apipattanavis 2007: 3-state Markov + daily joint kNN",
                         source="https://doi.org/10.1029/2006WR005714", seed=self.seed,
                         window_days=2 * self.window + 1, standardized_distance=int(self.standardize),
                         extreme_basis=self.extreme_basis, candidate_window_expansions=expansions,
                         transition_support_renormalizations=support_changes,
                         conditioning_limitation="Forecast weights apply to the bootstrap training sample; validate generated seasonal totals",
                         calendar_policy="Gregorian; February 29 omitted")
        return out


__all__ = ["ApipattanavisGenerator"]
