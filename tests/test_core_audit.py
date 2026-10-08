"""Regression checks for probability, rainfall and core API coherence."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.special import ndtri

from was_disaggregation.conditioning import (
    classify_seasons, normal_scores, tercile_normal_forecast,
    tercile_pdf_ratio_weights, yates_rank_probabilities,
)
from was_disaggregation.data import (
    canonicalize_observations, prepare_probabilities, season_dates, seasonal_cube, _validate_grid,
)
from was_disaggregation.model import WeatherGenerator
from was_disaggregation.multivariate import fit_multivariate, simulate_multivariate
from was_disaggregation.rainfall import fit_rainfall, simulate_rainfall, seasonal_total_moments
from was_disaggregation.spatial import IndependentField, fit_distance_model


def test_percentage_pdf_ratio_matches_fractions():
    totals = np.arange(1., 10.)[:, None]
    years = np.arange(2001, 2010)
    cats, _ = classify_seasons(totals, years, climatology=(2001, 2009))
    z = normal_scores(totals, years, climatology=(2001, 2009))
    p = np.array([[.2], [.3], [.5]])
    w, _ = tercile_pdf_ratio_weights(z, cats, p)
    w_percent, _ = tercile_pdf_ratio_weights(z, cats, p * 100)
    np.testing.assert_allclose(w_percent, w)
    np.testing.assert_allclose([(w * (cats == c)).sum(0) for c in range(3)], p)


def test_pdf_ratio_respects_climatology_fallback_for_empty_categories():
    categories = np.array([[0], [0], [2], [2]])
    zscores = np.array([[-1.], [-.2], [.2], [1.]])
    p = np.array([[.1], [.7], [.2]])
    climate, flags = tercile_pdf_ratio_weights(zscores, categories, p, empty_policy="climatology")
    renormalized, _ = tercile_pdf_ratio_weights(zscores, categories, p, empty_policy="renormalize")
    np.testing.assert_allclose(climate, .25)
    assert flags[0] & 1
    assert not np.allclose(renormalized, climate)
    with pytest.raises(ValueError, match="empty categories"):
        tercile_pdf_ratio_weights(zscores, categories, p, empty_policy="raise")


@pytest.mark.parametrize("p", [[1., 0., 0.], [0., 0., 1.], [0., 1., 0.], [.2, .3, .5]])
def test_extreme_forecast_normal_density_stays_finite(p):
    mu, sigma = tercile_normal_forecast(np.asarray(p)[:, None])
    assert np.isfinite(mu).all() and np.isfinite(sigma).all()
    assert (sigma > 0).all()


@pytest.mark.parametrize("mean", [.25, .9])
def test_trace_simulation_preserves_fitted_mean_and_moments(mean):
    fit = fit_rainfall(np.full((3, 1, 1), mean), [7], np.ones((3, 1)),
                       include_trace=True, persistence="independent")
    n = 10_000
    gaussian_quantiles = ndtri((np.arange(n) + .5) / n)[:, None]
    generated = simulate_rainfall(fit, [7], n, lambda *args: gaussian_quantiles)
    assert (generated < 1.).all() and (generated > 0.).all()
    expected_mean, expected_variance = seasonal_total_moments(fit, [7])
    np.testing.assert_allclose(generated.mean(axis=(0, 1)), [mean], atol=1e-12)
    np.testing.assert_allclose(expected_mean, [mean], atol=1e-12)
    np.testing.assert_allclose(generated.var(axis=(0, 1)), expected_variance, rtol=1e-6)


@pytest.mark.parametrize("rain", [0., 5.])
@pytest.mark.parametrize("prior", [0., 3.])
def test_spell_generator_respects_constant_dry_or_wet_records(rain, prior):
    fit = fit_rainfall(np.full((3, 10, 1), rain), np.full(10, 7), np.ones((3, 1)),
                       occurrence="spell", spell_prior=prior)
    np.testing.assert_allclose(fit.initial_wet, [float(rain >= 1)])
    relevant_hazard = fit.dry_hazard if rain == 0 else fit.wet_continue
    np.testing.assert_allclose(relevant_hazard, float(rain >= 1))
    rng = np.random.default_rng(12)
    generated = simulate_rainfall(fit, np.full(10, 7), 20,
                                  lambda n, step, stream: rng.normal(size=(n, 1)))
    assert np.isfinite(generated).all()
    assert (generated == 0).all() if rain == 0 else (generated >= 1).all()


@pytest.mark.parametrize("kwargs", [
    {"max_dry_run": 0}, {"max_wet_run": 1.5}, {"spell_prior": -1},
    {"initial_state": (np.zeros((3, 2)), np.ones((3, 2)))},
    {"initial_state": (np.zeros((3, 1)), np.zeros((3, 1)))},
])
def test_spell_fit_rejects_invalid_run_configuration(kwargs):
    with pytest.raises(ValueError):
        fit_rainfall(np.zeros((3, 10, 1)), np.full(10, 7), np.ones((3, 1)),
                     occurrence="spell", **kwargs)


def test_independent_class_draw_is_independent_across_sites():
    model = WeatherGenerator(spatial="independent", conditioning="mixture", class_draw="fitted")
    model.lat_ = np.array([5., 6.])
    model.prob_flat_ = np.full((3, 2), 1 / 3.)
    field = IndependentField(model.lat_, [0., 0.], seed=42, site_ids=[0, 1])
    classes = model._draw_classes(100, 0, field.sample)
    assert (classes[:, 0] != classes[:, 1]).sum() > 20


def test_shared_class_draw_and_member_batches_are_reproducible():
    model = WeatherGenerator(spatial="independent", conditioning="mixture", class_draw="shared")
    model.lat_ = np.array([5., 6.])
    model.prob_flat_ = np.full((3, 2), 1 / 3.)
    unused_draw = lambda *args: pytest.fail("shared draws must not request a site field")
    full = model._draw_classes(20, 0, unused_draw)
    batch = model._draw_classes(10, 10, unused_draw)
    np.testing.assert_array_equal(full[10:], batch)
    np.testing.assert_array_equal(full[:, 0], full[:, 1])


def test_precursor_spell_history_omits_leap_day():
    dates = pd.date_range("2004-01-01", "2004-03-01")
    rain = np.zeros((len(dates), 1, 1))
    rain[dates == pd.Timestamp("2004-02-29")] = 5
    obs = xr.Dataset({"PRCP": (("T", "Y", "X"), rain)},
                     coords={"T": dates, "Y": [5.], "X": [0.]})
    model = WeatherGenerator(months=(3,), occurrence="spell", max_dry_run=3)
    model.years_ = np.array([2004])
    wet, run = model._initial_state(obs, 1)
    np.testing.assert_array_equal(wet, [[0.]])
    assert run[0, 0] >= 4


def test_calendar_cube_keeps_cross_year_seasons_and_drops_incomplete_ones():
    dates = pd.date_range("2003-12-01", "2005-02-28")
    dates = dates[dates != pd.Timestamp("2004-12-02")]
    ds = xr.Dataset({"PRCP": (("T", "Y", "X"), np.ones((len(dates), 1, 1)))},
                    coords={"T": dates, "Y": [5.], "X": [0.]})
    cube = seasonal_cube(ds, (12, 1, 2))
    np.testing.assert_array_equal(cube.season_year, [2003])
    assert cube.sizes["day"] == 90
    assert "2004" in cube.attrs["dropped_incomplete_seasons"]
    assert len(season_dates(2003, (12, 1, 2))) == 90


def test_unit_conversion_and_probability_invalid_triples():
    ds = xr.Dataset({
        "PRCP": (("time", "lat", "lon"), np.full((3, 1, 1), 1 / 86400), {"units": "kg m-2 s-1"}),
        "TMIN": (("time", "lat", "lon"), np.full((3, 1, 1), 293.15), {"units": "K"}),
    }, coords={"time": pd.date_range("2001-01-01", periods=3), "lat": [5.], "lon": [0.]})
    converted = canonicalize_observations(ds)
    np.testing.assert_allclose(converted.PRCP.compute(), 1.)
    np.testing.assert_allclose(converted.TMIN.compute(), 20.)
    p = xr.DataArray(np.array([[20, 20], [30, np.nan], [50, 50]])[:, None, :],
                     dims=("probability", "Y", "X"),
                     coords={"probability": ["PB", "PN", "PA"], "Y": [5.], "X": [0., 1.]})
    prepared = prepare_probabilities(p)
    np.testing.assert_allclose(prepared.isel(X=0).values[:, 0], [.2, .3, .5])
    assert np.isnan(prepared.isel(X=1)).all()


def test_grid_validator_accepts_datetime_coordinate_on_dataarray():
    rain = xr.DataArray(np.ones((3, 1, 1)), dims=("T", "Y", "X"),
                        coords={"T": pd.date_range("2001-01-01", periods=3), "Y": [5.], "X": [0.]})
    _validate_grid(rain, time=True)


def test_dependence_fit_cannot_reuse_different_month_sequence():
    rng = np.random.default_rng(3)
    rain = np.ones((3, 30, 1))
    values = {"TMIN": rng.normal(size=rain.shape)}
    weights = np.ones((3, 1))
    months = np.repeat([7, 8], 15)
    fit = fit_multivariate(values, rain, months, weights)
    with pytest.raises(ValueError, match="same variables, months and sites"):
        fit_multivariate(values, rain, months[::-1], weights, dependence=fit)


def test_multivariate_curves_are_invariant_to_large_weight_rescaling():
    rng = np.random.default_rng(4)
    rain = np.ones((3, 30, 1))
    values = {"TMIN": rng.normal(size=rain.shape)}
    month = np.full(30, 7)
    fit = fit_multivariate(values, rain, month, np.ones((3, 1)))
    scaled = fit_multivariate(values, rain, month, np.full((3, 1), 1e308))
    np.testing.assert_allclose(scaled.mean, fit.mean)
    np.testing.assert_allclose(scaled.std, fit.std)


def test_mixture_missing_variable_masks_only_members_of_unsupported_class():
    rain = np.ones((3, 10, 1))
    temperature = np.full(rain.shape, 20.)
    temperature[0] = np.nan
    values, month = {"TMIN": temperature}, np.full(10, 7)
    base = fit_multivariate(values, rain, month, np.ones((3, 1)))
    fits = [fit_multivariate(values, rain, month, np.eye(3)[c, :, None], dependence=base)
            for c in range(3)]
    selected = fits[0].select(np.array([[0], [1], [2]]), fits[1:])
    generated = simulate_multivariate(selected, np.ones((3, 10, 1)), month,
                                      lambda n, step, stream: np.zeros((n, 1)))
    assert np.isnan(generated["TMIN"][0]).all()
    np.testing.assert_allclose(generated["TMIN"][1:], 20.)


def test_weather_generator_mixture_preserves_supported_variable_classes():
    dates = pd.DatetimeIndex(np.concatenate([season_dates(year, (7,)).values for year in (2001, 2002, 2003)]))
    rain = np.repeat([5., 10., 20.], 31)[:, None, None]
    temperature = np.full_like(rain, 20.)
    temperature[:31] = np.nan
    ds = xr.Dataset({
        "PRCP": (("T", "Y", "X"), rain, {"units": "mm d-1"}),
        "TMIN": (("T", "Y", "X"), temperature, {"units": "degC"}),
    }, coords={"T": dates, "Y": [5.], "X": [0.]})
    p = xr.DataArray(np.full((3, 1, 1), 1 / 3.), dims=("probability", "Y", "X"),
                     coords={"probability": ["PB", "PN", "PA"], "Y": ds.Y, "X": ds.X})
    model = WeatherGenerator(months=(7,), climatology=(2001, 2003), spatial="independent",
                              conditioning="mixture", mixture_shrinkage="none").fit(ds, p)
    generated = model.generate(2004, n_members=30)
    below = generated.tercile_class.values[:, 0, 0] == 0
    assert below.any() and (~below).any()
    assert np.isnan(generated.TMIN.values[below]).all()
    np.testing.assert_allclose(generated.TMIN.values[~below], 20.)


def test_shorter_mre_total_window_masks_only_unsupported_rainfall_classes():
    from was_disaggregation.attributes import SeasonalTotal
    from was_disaggregation.mre import SeasonalConstraint

    dates = pd.DatetimeIndex(np.concatenate([
        season_dates(year, (5, 6, 7, 8, 9, 10)).values for year in (2001, 2002, 2003)
    ]))
    rain = np.select([dates.year == 2001, dates.year == 2002], [5., 10.], default=20.)[:, None, None]
    rain[(dates.year == 2001) & (dates.month == 5)] = np.nan
    ds = xr.Dataset({"PRCP": (("T", "Y", "X"), rain, {"units": "mm d-1"})},
                     coords={"T": dates, "Y": [5.], "X": [0.]})
    p = xr.DataArray(np.full((3, 1, 1), 1 / 3.), dims=("probability", "Y", "X"),
                     coords={"probability": ["PB", "PN", "PA"], "Y": ds.Y, "X": ds.X})
    constraint = SeasonalConstraint(SeasonalTotal(window=("07-01", "09-30")))
    model = WeatherGenerator(months=(5, 6, 7, 8, 9, 10), climatology=(2001, 2003),
                              spatial="independent", conditioning="mixture", mixture_shrinkage="none",
                              weighting="mre", constraints=[constraint]).fit(ds, p)
    generated = model.generate(2004, n_members=30)
    labels = generated.tercile_class.values[:, 0, 0]
    supported = labels >= 0
    assert supported.any() and (~supported).any()
    assert set(labels[supported]) == {1, 2}
    assert np.isnan(generated.PRCP.values[~supported]).all()
    assert np.isfinite(generated.PRCP.values[supported]).all()


def test_binary_spatial_fit_rejects_invalid_values_even_without_usable_pairs():
    with pytest.raises(ValueError, match="requires occurrence values"):
        fit_distance_model(np.full((40, 1), 2.), [5.], [0.], transform="binary")


@pytest.mark.parametrize("kwargs", [{"seed": -1}, {"n_features": 0}, {"max_sites": 1.5},
                                      {"wet_threshold": np.nan}, {"months": (True,)}])
def test_generator_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        WeatherGenerator(**kwargs)


def test_generator_rejects_ignored_year_prior():
    prior = xr.DataArray([1.], dims="season_year", coords={"season_year": [2001]})
    with pytest.raises(ValueError, match="year_prior requires"):
        WeatherGenerator(year_prior=prior)


@pytest.mark.parametrize("kwargs", [{"n": 0}, {"n": 3, "strength": 0}, {"n": 3, "selection": .5}])
def test_rank_weights_reject_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        yates_rank_probabilities(**kwargs)


@pytest.mark.parametrize("conditioning", ["mean", "mixture"])
def test_generator_end_to_end_units_constraints_and_member_batching(conditioning):
    dates = pd.date_range("2001-01-01", "2006-12-31")
    rng = np.random.default_rng(123)
    shape = (len(dates), 2, 2)
    rain = rng.gamma(2., 5., size=shape) * (rng.random(shape) < .6)
    tmin = 20 + rng.normal(size=shape)
    ds = xr.Dataset({
        "PRCP": (("T", "Y", "X"), rain, {"units": "mm d-1"}),
        "TMIN": (("T", "Y", "X"), tmin, {"units": "degC"}),
        "TMAX": (("T", "Y", "X"), tmin + 5, {"units": "degC"}),
    }, coords={"T": dates, "Y": [5., 6.], "X": [0., 1.]})
    p = xr.DataArray(np.broadcast_to(np.array([.2, .3, .5])[:, None, None], (3, 2, 2)),
                     dims=("probability", "Y", "X"),
                     coords={"probability": ["PB", "PN", "PA"], "Y": ds.Y, "X": ds.X})
    model = WeatherGenerator(climatology=(2001, 2006), spatial="independent",
                              conditioning=conditioning, mixture_shrinkage=.5).fit(ds, p)
    full = model.generate(2007, n_members=4)
    batch = model.generate(2007, n_members=2, member_start=2)
    xr.testing.assert_equal(full.isel(member=slice(2, 4)), batch)
    assert full.PRCP.attrs["units"] == "mm d-1"
    assert full.TMIN.attrs["units"] == "degC"
    assert (full.TMIN <= full.TMAX).all()
    assert np.isfinite(full.PRCP).all()
    assert "was-disaggregation" in full.attrs["generator"]
    with pytest.raises(ValueError):
        model.generate(2007, n_members=1.5)
    with pytest.raises(ValueError):
        model.generate(2007.5)
