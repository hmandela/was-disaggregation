"""Descriptive weather diagnostics; these are not hindcast skill estimates.

The fixed historical climatology defines terciles. Ensemble frequencies measure
conditioning fidelity, never forecast reliability from one operational season.
NaNs remain missing and interrupt spells; sums require complete seasons.

References and scope
--------------------
Edwin B. Wilson (1927), "Probable Inference, the Law of Succession, and
Statistical Inference", Journal of the American Statistical Association 22,
209-212. DOI: 10.1080/01621459.1927.10502953.
    wilson_probability_interval: binomial score interval, requiring independent
    members; not a confidence interval for structural meteorological error.
William M. Briggs and Daniel S. Wilks (1996), "Extension of the Climate
Prediction Center Long-Lead Temperature and Precipitation Outlooks to General
Weather Statistics", Journal of Climate 9, 3496-3504.
DOI: 10.1175/1520-0442(1996)009<3496:EOTCPC>2.0.CO;2.
    Tercile diagnostics compare the target probabilities to observed/generated
    class masses used by seasonal reweighting. They do not reproduce the
    authors' outlook-verification experiment.

Grid-cell integration, finite-member selection and descriptive summaries are
package utilities. The exact spherical area uses sin(latitude) edge differences;
select_members reports historical-pool weights separately from realized
member frequencies. Neither guarantees new out-of-sample forecast skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import ndtri
from scipy.optimize import linprog

from .conditioning import classify_seasons
from .data import prepare_probabilities, seasonal_cube, season_dates


def grid_cell_area_weights(grid, *, longitude_periodic=None):
    """Normalized spherical cell areas inferred from rectilinear cell centres.

    Latitude edges are midpoint edges clipped to the poles. Regional longitude
    is unwrapped across the dateline; global periodicity can be explicit, or is
    inferred only when endpoint midpoint edges span 360 degrees. Bounds supplied
    by a data provider are preferable to inferred edges. Singleton axes have
    arbitrary one-degree width, which cancels when weights are normalized.
    """
    lat, lon = np.asarray(grid["Y"], float), np.asarray(grid["X"], float)
    if lat.ndim != 1 or lon.ndim != 1 or not lat.size or not lon.size or not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Y/X must be nonempty finite one-dimensional coordinates")
    if ((lat < -90) | (lat > 90)).any() or np.unique(lat).size != lat.size:
        raise ValueError("Latitudes must be unique and within [-90,90]")
    if np.unique(np.mod(lon, 360)).size != lon.size:
        raise ValueError("Longitude centres duplicate the same physical meridian")

    def edges(values):
        if values.size == 1:
            return np.array([values[0] - .5, values[0] + .5])
        delta = np.diff(values)
        if not ((delta > 0).all() or (delta < 0).all()):
            raise ValueError("Grid axes must be monotone after longitude unwrapping")
        return np.r_[values[0] - delta[0] / 2, (values[:-1] + values[1:]) / 2,
                     values[-1] + delta[-1] / 2]

    lat_edge = np.clip(edges(lat), -90, 90)
    lat_area = np.abs(np.diff(np.sin(np.deg2rad(lat_edge))))
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lon)))
    lon_edge = None if longitude_periodic is True else edges(unwrapped)
    if longitude_periodic is None:
        longitude_periodic = bool(lon.size > 1 and np.isclose(abs(lon_edge[-1] - lon_edge[0]), 360))
    if longitude_periodic:
        order = np.argsort(np.mod(lon, 360))
        sorted_lon = np.mod(lon[order], 360)
        gaps = np.diff(np.r_[sorted_lon, sorted_lon[0] + 360])
        widths = (gaps + np.roll(gaps, 1)) / 2
        lon_width = np.empty(lon.shape)
        lon_width[order] = widths
    else:
        lon_width = np.abs(np.diff(lon_edge))
        if lon_width.sum() > 360 + 1e-8:
            raise ValueError("Regional inferred longitude edges exceed 360 degrees; specify longitude_periodic=True")
    area = lat_area[:, None] * np.deg2rad(lon_width)[None]
    if area.sum() <= 0:
        raise ValueError("Grid has zero inferred area")
    return xr.DataArray(area / area.sum(), dims=("Y", "X"), coords={"Y": grid["Y"], "X": grid["X"]},
                        attrs={"definition": "normalized spherical area from midpoint cell edges", "longitude_periodic": int(bool(longitude_periodic))})


def wilson_probability_interval(count, total, confidence=.95):
    """Wilson binomial sampling interval, including zero observed categories.

    Requires independent exchangeable members; it measures Monte Carlo sampling
    precision, not meteorological predictive uncertainty or forecast skill.

    References
    ----------
    Edwin B. Wilson (1927), DOI: 10.1080/01621459.1927.10502953.
    See the module bibliography for the original statistical construction.
    """
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    k, n = np.broadcast_arrays(np.asarray(count, float), np.asarray(total, float))
    if (np.isfinite(k) & np.isfinite(n) & ((k < 0) | (n < k) | (n < 0))).any():
        raise ValueError("Need 0 <= count <= total")
    valid = np.isfinite(k) & np.isfinite(n) & (n > 0)
    p = np.divide(k, n, out=np.full(k.shape, np.nan), where=valid)
    n = np.where(valid, n, np.nan)
    z2 = ndtri((1 + confidence) / 2) ** 2
    denominator = 1 + z2 / np.where(valid, n, np.nan)
    centre = (p + z2 / (2 * n)) / denominator
    half = np.sqrt(z2 * (p * (1 - p) / n + z2 / (4 * n * n))) / denominator
    return np.maximum(centre - half, 0), np.minimum(centre + half, 1)


def seasonal_totals(precipitation: xr.DataArray, time_dim: str = "T") -> xr.DataArray:
    """Sum a single season, leaving any incompletely observed season missing."""
    if precipitation.sizes.get(time_dim, 0) == 0:
        raise ValueError("A nonempty season time dimension is required")
    result = precipitation.sum(time_dim, skipna=False)
    result.name = "seasonal_total"
    result.attrs.update(units="mm", missing_policy="requires every seasonal day")
    return result


def tercile_frequencies(totals: xr.DataArray, thresholds: xr.DataArray,
                       sample_dim: str = "member") -> xr.DataArray:
    """Frequencies using fixed q33/q67 thresholds; boundary values go lower.

    ``thresholds`` has a two-element ``threshold`` (or ``quantile``) dimension
    in ascending order.
    Missing samples are excluded; missing thresholds mask all three frequencies.
    """
    threshold_dim = "threshold" if "threshold" in thresholds.dims else "quantile"
    if thresholds.sizes.get(threshold_dim) != 2 or sample_dim not in totals.dims:
        raise ValueError("Need two thresholds and the requested sample dimension")
    lo = thresholds.isel({threshold_dim: 0}, drop=True)
    hi = thresholds.isel({threshold_dim: 1}, drop=True)
    if bool((hi < lo).any()):
        raise ValueError("tercile thresholds must be in ascending order")
    good = np.isfinite(totals) & np.isfinite(lo) & np.isfinite(hi)
    count = good.sum(sample_dim)
    frequencies = [((totals <= lo) & good).sum(sample_dim),
                   ((totals > lo) & (totals <= hi) & good).sum(sample_dim),
                   ((totals > hi) & good).sum(sample_dim)]
    result = xr.concat(frequencies, dim=xr.IndexVariable("probability", ["PB", "PN", "PA"])) / count.where(count > 0)
    result.name = "generated_probability"
    return result


def _contiguous(time: xr.DataArray) -> np.ndarray:
    values = np.asarray(time.values)
    if np.issubdtype(values.dtype, np.datetime64):
        days = values.astype("datetime64[D]")
        delta = np.diff(days)
        dates = pd.DatetimeIndex(days)
        # February 28 -> March 1 is adjacent on the package's no-leap policy.
        omitted_leap_day = ((delta == np.timedelta64(2, "D"))
                            & (dates[:-1].month == 2) & (dates[:-1].day == 28)
                            & (dates[1:].month == 3) & (dates[1:].day == 1))
        return (delta == np.timedelta64(1, "D")) | omitted_leap_day
    return np.ones(max(len(values) - 1, 0), dtype=bool)


def _lag1(values, contiguous, occurrence=False, wet_threshold=1.0):
    a = np.asarray(values, dtype=float)
    good = np.isfinite(a[:-1]) & np.isfinite(a[1:]) & contiguous
    if good.sum() < 3:
        return np.nan
    if occurrence:
        a = (a >= wet_threshold).astype(float)
    x, y = a[:-1][good], a[1:][good]
    x, y = x - x.mean(), y - y.mean()
    denominator = np.sqrt(np.sum(x*x) * np.sum(y*y))
    return float(np.sum(x*y) / denominator) if denominator > 0 else np.nan


def dry_spell_lengths(values, wet_threshold: float = 1.0, contiguous=None) -> np.ndarray:
    """Lengths of observed dry runs; missing days/date gaps split runs.

    Runs touching boundaries or missing data are censored lower bounds, not
    inferred full spells. No observed dry day returns an empty integer array.
    """
    a = np.asarray(values, dtype=float)
    if a.ndim != 1:
        raise ValueError("dry_spell_lengths takes one time series")
    if not np.isfinite(wet_threshold) or wet_threshold <= 0:
        raise ValueError("wet_threshold must be positive")
    if contiguous is None:
        contiguous = np.ones(max(len(a) - 1, 0), dtype=bool)
    contiguous = np.asarray(contiguous, dtype=bool)
    if contiguous.shape != (max(len(a) - 1, 0),):
        raise ValueError("contiguous must have one entry per adjacent pair")
    runs, length = [], 0
    for i, value in enumerate(a):
        if i > 0 and not contiguous[i-1] and length:
            runs.append(length)
            length = 0
        if np.isfinite(value) and value < wet_threshold:
            length += 1
        else:
            if length:
                runs.append(length)
                length = 0
    if length:
        runs.append(length)
    return np.asarray(runs, dtype=int)


def weather_statistics(precipitation: xr.DataArray, time_dim: str = "T",
                       wet_threshold: float = 1.0) -> xr.Dataset:
    """Per-season rainfall statistics, keeping all non-time dimensions.

    Statistics are computed per ensemble member or historical year; adjacent
    years are never concatenated by this function. Dask is supported, but each
    single seasonal time series is rechunked together when needed.
    """
    if not np.isfinite(wet_threshold) or wet_threshold <= 0:
        raise ValueError("wet_threshold must be positive")
    if time_dim not in precipitation.dims:
        raise ValueError(f"Missing time dimension {time_dim}")
    contiguous = _contiguous(precipitation[time_dim])
    valid = np.isfinite(precipitation)
    count = valid.sum(time_dim)
    wet = (precipitation >= wet_threshold) & valid
    def apply(func):
        return xr.apply_ufunc(func, precipitation, input_core_dims=[[time_dim]],
                              output_core_dims=[[]], vectorize=True, dask="parallelized",
                              dask_gufunc_kwargs={"allow_rechunk": True}, output_dtypes=[float])
    def longest(x):
        if not np.isfinite(x).any():
            return np.nan
        lengths = dry_spell_lengths(x, wet_threshold=wet_threshold, contiguous=contiguous)
        return float(lengths.max(initial=0))
    result = xr.Dataset({
        "seasonal_total": seasonal_totals(precipitation, time_dim),
        "wet_frequency": wet.sum(time_dim) / count.where(count > 0),
        "mean_wet_amount": precipitation.where(wet).mean(time_dim, skipna=True),
        "lag1_amount": apply(lambda x: _lag1(x, contiguous)),
        "lag1_occurrence": apply(lambda x: _lag1(x, contiguous, True, wet_threshold)),
        "max_dry_spell": apply(longest),
        "valid_days": count,
    })
    result.max_dry_spell.attrs.update(units="days", note="Boundary/missing-data spells are censored lower bounds")
    result.attrs["wet_threshold_mm_day"] = float(wet_threshold)
    return result


def ensemble_diagnostics(generated: xr.Dataset, observations: xr.Dataset,
                         probabilities: xr.DataArray, months=(7, 8, 9),
                         climatology=(1991, 2020), wet_threshold=1.0,
                         tercile_method="empirical", quantile_method=None) -> xr.Dataset:
    """Compare one generated season with fixed-reference historical seasons.

    Returns maps of seasonal tercile frequencies minus target probabilities,
    fixed thresholds, valid-member counts, and ensemble/reference mean weather
    statistics. The Monte Carlo standard error uses the generated frequency and
    only quantifies finite-member sampling under independent realizations.
    This function does not compute RPSS, forecast calibration, or forecast skill.
    """
    if "PRCP" not in generated or "PRCP" not in observations:
        raise ValueError("generated and observations must contain PRCP")
    if "member" not in generated.PRCP.dims:
        raise ValueError("generated PRCP needs member,T,Y,X dimensions")
    if generated.sizes.get("T", 0) == 0:
        raise ValueError("generated must contain one complete season")
    generated_dates = np.asarray(generated.T.values)
    if not np.issubdtype(generated_dates.dtype, np.datetime64):
        raise ValueError("generated T must use Gregorian datetime64 dates")
    first_year = int(generated.T.dt.year.isel(T=0))
    expected_dates = np.asarray(season_dates(first_year, months))
    if not np.array_equal(generated_dates.astype("datetime64[D]"), expected_dates.astype("datetime64[D]")):
        raise ValueError("generated must contain exactly one complete requested season, with February 29 omitted")
    # Strict alignment prevents an unnoticed comparison of different grids.
    for dim in ("Y", "X"):
        if not generated[dim].equals(observations[dim]):
            raise ValueError(f"generated and observations must use identical {dim} coordinates")
    cube = seasonal_cube(observations, months=months)
    quantile_method = generated.attrs.get("quantile_method", "linear") if quantile_method is None else quantile_method
    totals_hist = seasonal_totals(cube.PRCP, "day").transpose("season_year", "Y", "X")
    years = totals_hist.season_year.values
    shape = (totals_hist.sizes["Y"], totals_hist.sizes["X"])
    _, threshold_values = classify_seasons(totals_hist.values.reshape(len(years), -1), years,
                                           climatology=climatology, method=tercile_method, quantile_method=quantile_method)
    thresholds = xr.DataArray(threshold_values.reshape(2, *shape), dims=("threshold", "Y", "X"),
                              coords={"threshold": ["q33", "q67"], "Y": generated.Y, "X": generated.X})
    target = prepare_probabilities(probabilities, target=generated)
    forecast_valid = np.isfinite(target).all("probability")
    generated_stats = weather_statistics(generated.PRCP, wet_threshold=wet_threshold)
    reference = cube.sel(season_year=slice(climatology[0], climatology[1]))
    reference_stats = weather_statistics(reference.PRCP, time_dim="day", wet_threshold=wet_threshold)
    frequency = tercile_frequencies(generated_stats.seasonal_total, thresholds).where(forecast_valid)
    n = np.isfinite(generated_stats.seasonal_total).sum("member").where(forecast_valid)
    result = xr.Dataset({
        "target_probability": target,
        "generated_probability": frequency,
        "probability_error": frequency - target,
        "monte_carlo_standard_error": np.sqrt(frequency * (1 - frequency) / n.where(n > 0)),
        "thresholds": thresholds,
        "valid_members": n,
    })
    counts = frequency * n
    lower, upper = xr.apply_ufunc(wilson_probability_interval, counts, n,
                                 output_core_dims=[[], []], dask="parallelized", output_dtypes=[float, float])
    result["monte_carlo_probability_lower"] = lower
    result["monte_carlo_probability_upper"] = upper
    for name in generated_stats:
        result["generated_" + name] = generated_stats[name].mean("member", skipna=True).where(forecast_valid)
        result["reference_" + name] = reference_stats[name].mean("season_year", skipna=True).where(forecast_valid)
    result.attrs.update({"purpose": "conditioning fidelity and descriptive weather realism; not hindcast skill",
                         "climatology_start": int(climatology[0]), "climatology_end": int(climatology[1]),
                         "tercile_method": tercile_method,
                         "quantile_method": quantile_method,
                         "tercile_boundaries": "PB <= q33; q33 < PN <= q67; PA > q67",
                         "reference_selection": "fixed climatology only; complete seasonal totals required",
                         "monte_carlo_note": "Frequency-based standard errors can be zero for unobserved categories; they are not forecast uncertainty bounds."})
    result.attrs["monte_carlo_interval"] = "95% Wilson sampling interval; assumes independent members, not forecast uncertainty"
    return result


def _attribute_classes(con, generated, observations, probabilities, climatology, tercile_method, quantile_method=None):
    """Historical thresholds, generated classes (member, site) and targets (3, site)."""
    from .mre import relabel_probabilities, thresholds_for, classify_fixed
    for dim in ("Y", "X"):
        if not generated[dim].equals(observations[dim]):
            raise ValueError(f"generated and observations must use identical {dim} coordinates")
    obs_years = list(range(int(climatology[0]), int(climatology[1]) + 1))
    year = int(generated.T.dt.year.isel(T=0))
    hist = con.attribute.compute(observations["PRCP"], obs_years).transpose("season_year", "Y", "X")
    ny, nx = hist.sizes["Y"], hist.sizes["X"]
    thr = thresholds_for(con, target=generated, n_sites=ny * nx)
    if thr is None:
        quantile_method = generated.attrs.get("quantile_method", "linear") if quantile_method is None else quantile_method
        _, thr = classify_seasons(hist.values.reshape(len(obs_years), -1), np.asarray(obs_years),
                                  climatology=climatology, method=tercile_method, quantile_method=quantile_method)
    gen = con.attribute.compute(generated["PRCP"], [year]).isel(season_year=0).transpose("member", "Y", "X")
    if not np.isfinite(gen.values).any():
        raise ValueError(f"{con.name}: attribute undefined on generated data; does the generated season "
                         f"cover {con.attribute.span()}?")
    g = gen.values.reshape(gen.sizes["member"], -1)
    source = con.probabilities if con.probabilities is not None else probabilities
    if source is None:
        raise ValueError(f"{con.name}: no probabilities")
    target = prepare_probabilities(relabel_probabilities(source), target=generated)
    target = target.transpose("probability", "Y", "X").values.reshape(3, -1)
    hist_cats = classify_fixed(hist.values.reshape(len(obs_years), -1), thr)
    return {"hist": hist, "thresholds": thr, "values": g, "classes": classify_fixed(g, thr),
            "historical_event_observed": hist.coords.get("event_observed"),
            "generated_event_observed": gen.coords.get("event_observed"),
            "target": target, "shape": (ny, nx), "climatological": np.stack(
                [np.divide((hist_cats == k).sum(0), (hist_cats >= 0).sum(0),
                           out=np.full(ny * nx, np.nan), where=(hist_cats >= 0).sum(0) > 0) for k in range(3)])}


def attribute_diagnostics(generated: xr.Dataset, observations: xr.Dataset, constraints,
                          probabilities: xr.DataArray | None = None, climatology=(1991, 2020),
                          tercile_method="empirical", quantile_method=None) -> xr.Dataset:
    """Class fidelity of every constrained season attribute (0.3.0).

    For each :class:`~was_disaggregation.mre.SeasonalConstraint` the attribute is
    computed on the historical reference years (terciles, or the constraint's
    fixed thresholds) and on every generated member. Generated class
    frequencies are compared with the target probabilities (``probabilities``
    for a constraint whose own probabilities are None) and with the observed
    climatological class frequencies, which differ from 1/3 when attribute
    values are tied (e.g. dry-spell lengths in whole days). The generated
    season must cover each attribute's data span.
    """
    rows = {k: [] for k in ("target_probability", "generated_probability", "probability_error",
                            "climatological_probability", "generated_median", "observed_median",
                            "generated_event_observed_frequency", "observed_event_observed_frequency")}
    names = []
    for con in constraints:
        r = _attribute_classes(con, generated, observations, probabilities, climatology, tercile_method, quantile_method)
        ny, nx = r["shape"]
        c = r["classes"]
        n = (c >= 0).sum(0)
        freq = np.stack([(c == k).sum(0) for k in range(3)]).astype(float)
        freq = np.divide(freq, n, out=np.full(freq.shape, np.nan), where=n > 0)
        names.append(con.name)
        rows["target_probability"].append(r["target"].reshape(3, ny, nx))
        rows["generated_probability"].append(freq.reshape(3, ny, nx))
        rows["probability_error"].append((freq - r["target"]).reshape(3, ny, nx))
        rows["climatological_probability"].append(r["climatological"].reshape(3, ny, nx))
        rows["generated_median"].append(np.nanmedian(r["values"], axis=0).reshape(ny, nx))
        rows["observed_median"].append(np.nanmedian(r["hist"].values, axis=0))
        for field, key, dim in (("generated_event_observed_frequency", "generated_event_observed", "member"),
                                ("observed_event_observed_frequency", "historical_event_observed", "season_year")):
            indicator = r[key]
            rows[field].append(indicator.mean(dim, skipna=True).values if indicator is not None else np.full((ny, nx), np.nan))
    if not names:
        raise ValueError("constraints must be nonempty")
    coords = {"constraint": names, "probability": ["PB", "PN", "PA"], "Y": generated.Y, "X": generated.X}
    ds = xr.Dataset({k: (("constraint", "probability", "Y", "X") if v[0].ndim == 3 else ("constraint", "Y", "X"),
                         np.stack(v)) for k, v in rows.items()}, coords=coords)
    ds.attrs.update(purpose="conditioning fidelity of constrained attributes; not forecast skill",
                    class_order="PB/PN/PA = low/normal/high value (early/normal/late, short/normal/long)",
                    climatology=f"{climatology[0]}-{climatology[1]}")
    return ds


def select_members(generated: xr.Dataset, observations: xr.Dataset, constraints, n_members: int,
                   probabilities: xr.DataArray | None = None, climatology=(1991, 2020),
                   tercile_method="empirical", method="mre", tolerance=None, seed=0,
                   area_weighted=True, constraint_mode="soft", longitude_periodic=None, quantile_method=None):
    """Post-hoc constrained selection of whole synthetic members (0.4.0).

    Generate a large pool (e.g. 5-10x the wanted members) with any generator,
    then pick ``n_members`` whole members whose attributes (total, onset, dry
    spells ...) match all forecasts. This is Croley / minimum-relative-entropy
    trace weighting applied to synthetic seasons. Each pool member receives a weight solving the
    area-mean constraints (features: fraction of cells of the member in each
    class). Members are then drawn by systematic resampling. Members stay whole
    space-time fields, so per-cell fidelity is exact only for the domain mean
    of a heterogeneous forecast; run it on a coherent region.

    Returns (selected Dataset with coord ``pool_member``, info dict).
    """
    from .mre import solve_weights
    nm = generated.sizes["member"]
    if nm < 1 or not isinstance(n_members, (int, np.integer)) or n_members < 1:
        raise ValueError("The pool and n_members must be nonempty/positive")
    feats, targets, taus, names, records = [], [], [], [], []
    area = None
    eligible = np.ones(nm, dtype=bool)
    for con in constraints:
        r = _attribute_classes(con, generated, observations, probabilities, climatology, tercile_method, quantile_method)
        ny, nx = r["shape"]
        if area is None:
            area = grid_cell_area_weights(generated, longitude_periodic=longitude_periodic).values.ravel() if area_weighted else np.ones(ny * nx)
        c = r["classes"]
        valid = np.isfinite(r["target"]).all(0) & (c >= 0).any(0)
        a = np.where(valid, area, 0.)
        if a.sum() <= 0:
            raise ValueError(f"{con.name}: no positive-area cells have valid forecasts and attributes")
        a = a / a.sum()
        # A member with missing attributes cannot be interpreted as a normal
        # class or normalized over a smaller, member-specific geographic area.
        eligible &= (c[:, a > 0] >= 0).all(axis=1)
        for k in (0, 2):
            feats.append(((c == k) * a).sum(1))
            targets.append(float(np.nansum(np.where(valid, r["target"][k], 0) * a)))
            taus.append(tolerance if tolerance is not None else con.tolerance)
        names.append(con.name)
        records.append((c, r["target"], a, valid))
    if not names:
        raise ValueError("constraints must be nonempty")
    if not eligible.any():
        raise ValueError("No pool member has complete constrained attributes on the forecast-valid region")
    g = np.stack(feats, -1)[:, None, :]
    wanted = np.asarray(targets)
    support_features = g[eligible, 0]
    feasibility = linprog(np.zeros(eligible.sum()),
                          A_eq=np.vstack([np.ones(eligible.sum()), support_features.T]),
                          b_eq=np.r_[1., wanted], bounds=(0, None), method="highs")
    w, info = solve_weights(g, wanted[None], eligible[:, None].astype(float), np.asarray(taus),
                           method=method, constraint_mode=constraint_mode)
    w = w[:, 0]
    rng = np.random.default_rng(seed)
    positions = (rng.random() + np.arange(n_members)) / n_members          # systematic resampling
    idx = np.minimum(np.searchsorted(np.cumsum(w), positions, side="right"), nm - 1)
    rng.shuffle(idx)
    out = generated.isel(member=idx).assign_coords(member=np.arange(n_members), pool_member=("member", idx))
    achieved = info["achieved"][0]
    report = {"names": names, "target_area_mean": np.asarray(targets).reshape(-1, 2),
              "achieved_area_mean": achieved.reshape(-1, 2), "pool_effective_members": float(info["effective_years"][0]),
              "distinct_members_selected": int(np.unique(idx).size), "weights": w}
    report.update(weighting_object="synthetic whole space-time members, not historical years",
                  constraint_domain="area-mean category frequencies; no per-cell guarantee",
                  constraint_mode=constraint_mode, eligible_members=int(eligible.sum()),
                  exact_joint_area_feasible=bool(feasibility.success),
                  feasibility_message=feasibility.message,
                  feature_support_min=support_features.min(0).reshape(-1, 2),
                  feature_support_max=support_features.max(0).reshape(-1, 2),
                  selected_area_mean=g[idx, 0].mean(0).reshape(-1, 2))
    weighted, selected, supported, unsupported, target_maps = [], [], [], [], []
    for c, target, a, valid in records:
        available = np.stack([(c[eligible] == k).any(0) for k in range(3)])
        wp = np.stack([np.sum(w[:, None] * (c == k), axis=0) for k in range(3)])
        sp = np.stack([(c[idx] == k).mean(0) for k in range(3)])
        weighted.append(np.where(valid, wp, np.nan).reshape(3, ny, nx))
        selected.append(np.where(valid, sp, np.nan).reshape(3, ny, nx))
        supported.append(available.reshape(3, ny, nx))
        unsupported.append((valid[None] & (target > 0) & ~available).reshape(3, ny, nx))
        target_maps.append(target.reshape(3, ny, nx))
    report.update(weighted_probability=np.stack(weighted), selected_probability=np.stack(selected),
                  target_probability=np.stack(target_maps), supported_categories=np.stack(supported),
                  unsupported_positive_target=np.stack(unsupported),
                  empirical_selection_error=report["selected_area_mean"] - report["achieved_area_mean"])
    out.attrs.update(member_selection=f"{method} selection of {n_members} from {nm} pool members on {', '.join(names)}",
                     pool_effective_members=round(report["pool_effective_members"], 1))
    out.attrs.update(member_selection_domain=report["constraint_domain"],
                     exact_joint_area_feasible=int(feasibility.success))
    return out, report
