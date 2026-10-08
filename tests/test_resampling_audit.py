"""Regressions for calendar, missing-value and constrained-resampling errors."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from was_disaggregation.attributes import CessationDate, MaxDrySpell, OnsetDate, SeasonalTotal
from was_disaggregation.data import season_dates
from was_disaggregation.diagnostics import attribute_diagnostics, select_members, weather_statistics
from was_disaggregation.mre import SeasonalConstraint, constrained_year_weights, solve_weights
from was_disaggregation.nonparametric import ForecastSchaakeGenerator, schaake_shuffle
from was_disaggregation.validation import (
    HindcastExperiment, crps_ensemble, daily_scores, ensemble_normal_scores,
    longest_dry_spell, model_tercile_probabilities, training_years,
)


def field(values, dates):
    return xr.DataArray(np.asarray(values, float).reshape(len(dates), 1, 1),
                        dims=("T", "Y", "X"), coords={"T": dates, "Y": [8.], "X": [2.]}, name="PRCP")


def probabilities():
    return xr.DataArray(np.array([1 / 3] * 3).reshape(3, 1, 1), dims=("probability", "Y", "X"),
                        coords={"probability": ["PB", "PN", "PA"], "Y": [8.], "X": [2.]})


def test_attributes_share_generator_no_leap_calendar():
    dates = pd.date_range("1999-12-01", "2000-02-29")
    rain = np.ones(len(dates))
    rain[-1] = 100.
    historical = field(rain, dates)
    generated = field(np.ones(90), season_dates(1999, (12, 1, 2)))
    attribute = SeasonalTotal(("12-01", "02-28"))
    assert attribute.compute(historical, [1999]).item() == 90.
    assert attribute.compute(generated, [1999]).item() == 90.

    dates = pd.date_range("2000-02-01", "2000-03-01")
    rainfall = np.ones(len(dates))
    rainfall[-2] = 100.
    assert SeasonalTotal(("02-01", "03-01")).compute(field(rainfall, dates), [2000]).item() == 29.

    dates = pd.date_range("2000-02-28", "2000-03-02")
    rainfall = field([0., 100., 20., 0.], dates)
    onset = OnsetDate(search=("02-28", "03-02"), accumulation_days=1, check_days=0)
    assert onset.compute(rainfall, [2000]).item() == 1.
    dry = MaxDrySpell(window=("02-28", "03-02"))
    assert dry.compute(field([0., 100., 0., 0.], dates), [2000]).item() == 3.
    # Explicit leap-day omission cannot split a synthetic dry spell.
    stats = weather_statistics(field([0., 0., 0.], pd.to_datetime(["2000-02-28", "2000-03-01", "2000-03-02"])))
    assert stats.max_dry_spell.item() == 3.


def test_onset_requires_only_the_needed_lookahead():
    rainfall = field([10., 10., 10., 1., 1.], pd.date_range("2001-05-01", periods=5))
    onset = OnsetDate(search=("05-01", "05-01"), accumulation_days=3, check_days=2)
    assert onset.span()[2] == 4
    assert onset.compute(rainfall, [2001]).item() == 0.


def test_main_precipitation_forecast_cannot_be_reinterpreted_as_onset_forecast():
    with pytest.raises(ValueError, match="requires SeasonalTotal"):
        SeasonalConstraint(OnsetDate())
    assert SeasonalConstraint(OnsetDate(), probabilities()).name == "onset"


def test_cross_year_cessation_offsets_start_at_search():
    dates = pd.date_range("2000-10-01", "2001-01-03")
    rainfall = field(np.zeros(len(dates)), dates)
    cessation = CessationDate(search=("01-01", "01-03"), balance_start="10-01")
    assert cessation.compute(rainfall, [2000]).item() == 0.


def test_missing_forecasts_never_become_probabilities_or_normal_scores():
    target = np.array([[np.nan, 2.]])
    train = np.array([[1., np.nan], [2., np.nan], [3., np.nan]])
    assert np.isnan(model_tercile_probabilities(target, train)).all()
    assert np.isnan(ensemble_normal_scores(target, train)).all()
    assert np.isnan(longest_dry_spell(np.full((3, 1), np.nan))).all()
    assert crps_ensemble([[1.], [np.inf]], [2.]).item() == 1.


def test_sorted_crps_matches_pairwise_definition_with_missing_members():
    ens = np.array([[0., 3., np.nan], [2., np.nan, np.nan], [4., 5., np.nan]])
    obs = np.array([1., 4., 2.])
    expected = []
    for site in range(ens.shape[1]):
        e = ens[np.isfinite(ens[:, site]), site]
        expected.append(np.abs(e - obs[site]).mean() - .5 * np.abs(e[:, None] - e[None]).mean() if len(e) else np.nan)
    np.testing.assert_allclose(crps_ensemble(ens, obs), expected, equal_nan=True)


def test_daily_scores_one_site_and_missing_denominators():
    obs = np.array([[2.], [4.], [np.nan]])
    ens = np.broadcast_to(obs, (2, *obs.shape)).copy()
    result = daily_scores(obs, ens)
    assert result["wet-day freq sim"] == 1.
    assert result["CRPS"] == 0.
    assert np.isnan(result["site corr obs"])
    assert np.isnan(result["site corr sim"])


def test_spatial_correlations_use_only_common_observations():
    obs = np.array([[1., 1.], [2., 2.], [100., np.nan]])
    result = daily_scores(obs, np.broadcast_to(obs, (2, *obs.shape)).copy())
    assert result["site corr obs"] == pytest.approx(1.)


@pytest.mark.parametrize("method", ["mre", "croley"])
def test_constrained_solver_support_normalization_and_convergence(method):
    features = np.array([[[0.]], [[1.]], [[0.]]])
    weights, info = solve_weights(features, [[.8]], [[1.], [1.], [0.]], [.01], method=method)
    assert weights.sum() == pytest.approx(1.)
    assert weights[-1, 0] == 0.
    assert info["achieved"][0, 0] == pytest.approx(.8, abs=.001)
    assert info["converged"].all()
    with pytest.raises(ValueError, match="positive tolerances"):
        solve_weights(features, [[.8]], [[1.], [1.], [0.]], [0.], method=method)
    with pytest.raises(ValueError, match="finite"):
        solve_weights(features * np.nan, [[.8]], [[1.], [1.], [0.]], [.01], method=method)


def test_empty_prior_reports_no_history():
    _, flags, info = constrained_year_weights(
        {"total": np.arange(6, dtype=float)[:, None]}, {"total": np.full((3, 1), 1 / 3)},
        np.arange(2000, 2006), climatology=(2000, 2005), prior=np.zeros(6),
    )
    assert flags[0] & 4
    assert info["effective_years"][0] == 0.


def july_observations():
    dates = np.concatenate([np.asarray(season_dates(y, (7,))) for y in (1991, 1992, 1993)])
    rain = field(np.repeat([1., 2., 3.], 31), dates)
    solar = np.r_[np.ones(62), 42., np.full(30, np.nan)]
    return xr.Dataset({"PRCP": rain, "SOLAR": field(solar, dates)})


def test_schaake_exact_redraw_preserves_rare_valid_donor_support():
    prior = xr.DataArray([0., 0., 1.], dims="season_year", coords={"season_year": [1991, 1992, 1993]})
    model = ForecastSchaakeGenerator(months=(7,), climatology=(1991, 1993), window=30)
    model.fit(july_observations(), probabilities(), year_prior=prior)
    generated = model.generate(2025, n_members=2)
    assert np.isfinite(generated.to_array()).all()
    assert (generated.PRCP == 3.).all()
    assert (generated.SOLAR == 42.).all()
    assert generated.attrs["cells_masked_unresolved_missing"] == 0


def test_schaake_rejects_zero_prior_and_unweighted_template_support():
    prior = xr.DataArray([0., 0., 0.], dims="season_year", coords={"season_year": [1991, 1992, 1993]})
    model = ForecastSchaakeGenerator(months=(7,), climatology=(1991, 1993))
    with pytest.raises(ValueError, match="positive donor mass"):
        model.fit(july_observations(), probabilities(), year_prior=prior)
    prior[-1] = 1.
    model = ForecastSchaakeGenerator(months=(7,), climatology=(1991, 1993), template_years="weighted")
    with pytest.raises(ValueError, match="complete Schaake template"):
        model.fit(july_observations(), probabilities(), year_prior=prior)


def test_select_members_excludes_missing_attribute_pool_members():
    obs = july_observations()[["PRCP"]]
    dates = season_dates(2025, (7,))
    pool = field(np.ones(31), dates).expand_dims(member=[0, 1, 2]).copy()
    pool.values[0] = np.nan
    pool.values[2] *= 3.
    constraint = SeasonalConstraint(SeasonalTotal(("07-01", "07-31")), thresholds=[40., 70.])
    selected, info = select_members(xr.Dataset({"PRCP": pool}), obs, [constraint], 10,
                                    probabilities=probabilities(), climatology=(1991, 1993))
    assert info["weights"][0] == 0.
    assert not (selected.pool_member == 0).any()
    with pytest.raises(ValueError, match="positive"):
        select_members(xr.Dataset({"PRCP": pool}), obs, [constraint], 0, probabilities=probabilities())


def test_hindcast_requires_target_calendar_grid_and_training_years():
    obs = july_observations()[["PRCP"]]
    experiment = HindcastExperiment(obs, [1991, 1992, 1993], (7,), {"total": SeasonalTotal(("07-01", "07-31"))})
    wrong = obs.PRCP.sel(T=slice("1991-07-01", "1991-07-31")).expand_dims(member=[0])
    with pytest.raises(ValueError, match="T coordinates"):
        experiment._as_dataarray(wrong, 1992)
    with pytest.raises(ValueError, match="nonnegative"):
        training_years([2000, 2001], 2000, buffer=-1)


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_hindcast_runs_without_fork_on_windows_and_macos(monkeypatch, platform):
    import was_disaggregation.validation as validation

    def no_fork(*args, **kwargs):
        raise AssertionError("fork must not be used on this platform")

    monkeypatch.setattr(validation.sys, "platform", platform)
    monkeypatch.setattr(validation.mp, "get_context", no_fork)
    obs = july_observations()[["PRCP"]]
    experiment = HindcastExperiment(obs, [1991, 1992, 1993], (7,),
                                    {"total": SeasonalTotal(("07-01", "07-31"))})

    def predict(train, year):
        return np.ones((2, 31, 1)) * (year - 1990)

    with pytest.warns(RuntimeWarning, match="sequential folds"):
        experiment.add("basic", predict).run(n_jobs=2, verbose=False)
    assert set(experiment.results_["basic"]) == {1991, 1992, 1993}


def test_attribute_diagnostics_rejects_misaligned_grids():
    obs = july_observations()[["PRCP"]]
    generated = xr.Dataset({"PRCP": field(np.ones(31), season_dates(2025, (7,))).expand_dims(member=[0])})
    generated = generated.assign_coords(X=[3.])
    con = SeasonalConstraint(SeasonalTotal(("07-01", "07-31")))
    with pytest.raises(ValueError, match="identical X coordinates"):
        attribute_diagnostics(generated, obs, [con], probabilities(), climatology=(1991, 1993))


def test_hindcast_correlation_uses_paired_valid_years():
    frame = pd.DataFrame({"obs": [1., 2., np.nan], "ens_mean": [2., 4., 6.],
                          "spread": [1., 1., 1.], "err2": [1., 4., np.nan]})
    assert HindcastExperiment._skill(frame)["corr"] == pytest.approx(1.)


def test_hindcast_forecast_reference_and_domain_means_share_valid_sites():
    one = july_observations().PRCP
    obs = xr.concat([one, one * 100.], dim="X").assign_coords(X=[2., 3.]).transpose("T", "Y", "X")
    attributes = {"total": SeasonalTotal(("07-01", "07-31"))}
    full = HindcastExperiment(obs, [1991, 1992, 1993], (7,), attributes)
    single = HindcastExperiment(one, [1991, 1992, 1993], (7,), attributes)

    def predict_single(train, year):
        return np.ones((2, 31, 1)) * (year - 1990)

    def predict_full(train, year):
        return np.concatenate([predict_single(train, year), np.full((2, 31, 1), np.nan)], axis=2)

    full.add("perfect", predict_full).run(verbose=False)
    single.add("perfect", predict_single).run(verbose=False)
    columns = ["RPS", "RPS_clim", "CRPS", "CRPS_clim", "obs", "ens_mean"]
    np.testing.assert_allclose(full.yearly()[columns], single.yearly()[columns], equal_nan=True)
    assert full.scores(n_boot=3).loc[("perfect", "total"), "RPSS"] == 1.


def test_schaake_tie_breaking_keeps_exact_marginals_and_missing_columns():
    values = np.array([[1., np.nan], [3., np.nan], [2., np.nan]])
    template = np.array([[0., np.nan], [0., np.nan], [1., np.nan]])
    shuffled = schaake_shuffle(values, template, ties="random", rng=np.random.default_rng(0))
    np.testing.assert_array_equal(np.sort(shuffled[:, 0]), [1., 2., 3.])
    assert np.isnan(shuffled[:, 1]).all()
