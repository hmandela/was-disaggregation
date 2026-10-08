"""Reproductions for missingness, calendar and API defects found during audit."""
import numpy as np
from itertools import product
import pandas as pd
import pytest
import xarray as xr

from was_disaggregation.data import season_dates
from was_disaggregation.dynamical import (
    DailyBiasCorrector, DynamicalDownscaler, NHMM, NHSMM,
    _padded_dates, ensemble_copula_coupling, preferential_dates,
    season_blocks, stratified_quantiles, synthetic_model_ensemble,
)
from was_disaggregation.glm import GLMWeatherGenerator


def observations(start="1999-01-01", stop="2003-12-31", constant_continuous=False):
    dates = pd.date_range(start, stop)
    rng = np.random.default_rng(123)
    rainfall = rng.gamma(2., 4., (len(dates), 1, 2))
    rainfall[rng.random(rainfall.shape) < .45] = 0.
    ds = xr.Dataset({"PRCP": (("T", "Y", "X"), rainfall)},
                    coords={"T": dates, "Y": [9.], "X": [2., 3.]})
    ds.PRCP.attrs["units"] = "mm d-1"
    if constant_continuous:
        for name in ("TMIN", "TMAX"):
            ds[name] = xr.zeros_like(ds.PRCP)
            ds[name].attrs["units"] = "degC"
    return ds


@pytest.fixture(scope="module")
def fitted_glm():
    return GLMWeatherGenerator(climatology=(1999, 2003), harmonics=1,
        spatial="independent", calibrate_covariate=False).fit(observations())


def test_season_blocks_rejects_missing_dates_and_duplicate_days():
    ds = observations("2001-07-01", "2001-09-30")
    with pytest.raises(ValueError, match="missing daily dates"):
        season_blocks(ds.PRCP.isel(T=slice(1, None)), [2001], (7, 8, 9))
    duplicate = xr.concat([ds.PRCP, ds.PRCP.isel(T=[0])], dim="T")
    with pytest.raises(ValueError, match="Duplicate|unique daily"):
        season_blocks(duplicate, [2001], (7, 8, 9))


def test_leap_padded_templates_keep_month_day_alignment():
    dates = _padded_dates(1999, (12, 1, 2), 3)
    assert len(dates) == 90 + 6
    assert not ((dates.month == 2) & (dates.day == 29)).any()
    assert dates[3:93].equals(season_dates(1999, (12, 1, 2)))
    assert dates[-3:].strftime("%m-%d").tolist() == ["03-01", "03-02", "03-03"]


@pytest.mark.parametrize("method", ["qm", "loci_qm"])
def test_bias_corrector_never_converts_missing_calibration_to_dry(method):
    model = np.ones((2, 3, 10, 2))
    obs = np.zeros((2, 10, 2))
    obs[..., 1] = np.nan
    month = np.full(10, 7)
    corrector = DailyBiasCorrector(method, n_quantiles=10).fit(model, obs, month)
    out = corrector.transform(model[0], month)
    assert (out[..., 0] == 0).all()
    assert np.isnan(out[..., 1]).all()


def test_stratified_quantiles_preserves_finite_mean_and_missing_columns():
    sample = np.array([[1., np.nan, 10.], [2., np.nan, np.nan], [3., np.nan, 30.]])
    out = stratified_quantiles(sample, 5)
    np.testing.assert_allclose(out[:, [0, 2]].mean(0), [2., 20.])
    assert np.isnan(out[:, 1]).all()


@pytest.mark.parametrize("method", ["Q", "R"])
def test_ecc_keeps_masked_sites_and_rejects_partial_raw_ranks(method):
    raw = np.array([[2., np.nan], [1., np.nan]])
    cal = np.array([[3., np.nan], [4., np.nan], [5., np.nan]])
    out = ensemble_copula_coupling(raw, cal, method, np.random.default_rng(5))
    assert np.isnan(out[:, 1]).all()
    assert out[0, 0] >= out[1, 0]
    raw[0, 0] = np.nan
    with pytest.raises(ValueError, match="template.*finite"):
        ensemble_copula_coupling(raw, cal, method, np.random.default_rng(5))


def test_preferential_templates_ignore_masked_fields_and_fill_member_count():
    forecast = np.array([[1., np.nan], [2., np.nan]])
    templates = np.array([[[1., np.nan], [2., np.nan]],
                          [[3., np.nan], [4., np.nan]]])
    yi, oi, score = preferential_dates(forecast, templates, 5, window=0)
    assert yi.size == oi.size == score.size == 5
    assert yi[0] == 0
    assert np.isfinite(score).all()


@pytest.mark.parametrize("cls", [NHMM, NHSMM])
def test_occurrence_only_models_do_not_simulate_unfitted_amounts(cls):
    rain = np.zeros((2, 20, 2))
    rain[:, ::2, 0] = 8.
    rain[..., 1] = np.nan
    predictors = np.zeros((2, 20, 1))
    kwargs = {"max_duration": 3, "init_iter": 0} if cls is NHSMM else {}
    model = cls(n_states=2, amounts=False, n_iter=2, **kwargs).fit(rain, predictors)
    out, states = model.simulate(predictors, n_sim=3)
    assert states.shape == (6, 20)
    assert set(np.unique(out[..., 0])).issubset({0., 1.})
    assert np.isnan(out[..., 1]).all()
    assert np.isfinite(model.loglik_).all()
    if cls is NHSMM:
        assert model.loglik_[-1] == pytest.approx(model.loglik(rain, predictors))
    else:
        le, lt = model._log_emission(rain), model._trans(predictors)
        gamma, _, ll = model._forward_backward(le, lt)
        np.testing.assert_allclose(model.gamma_, gamma)
        assert model.loglik_[-1] == pytest.approx(ll)


def test_nhmm_inference_matches_enumerated_paths():
    model = NHMM(n_states=2)
    model.pi0_ = np.array([.4, .6])
    emission = np.array([[.8, .2], [.3, .7], [.6, .4]])
    transition = np.array([[.75, .25], [.2, .8]])
    probabilities = []
    paths = list(product(range(2), repeat=3))
    for path in paths:
        prob = model.pi0_[path[0]] * emission[0, path[0]]
        for day in range(1, 3):
            prob *= transition[path[day - 1], path[day]] * emission[day, path[day]]
        probabilities.append(prob)
    probabilities = np.asarray(probabilities)
    expected = np.zeros((3, 2))
    for path, prob in zip(paths, probabilities):
        for day, state in enumerate(path):
            expected[day, state] += prob / probabilities.sum()
    gamma, _, ll = model._forward_backward(np.log(emission)[None],
        np.broadcast_to(np.log(transition), (1, 3, 2, 2)))
    np.testing.assert_allclose(gamma[0], expected)
    assert ll == pytest.approx(np.log(probabilities.sum()))


def test_nhsmm_inference_matches_expanded_chain_paths():
    model = NHSMM(n_states=2, max_duration=2)
    initial = np.array([.2, .3, .1, .4])
    hazard = np.array([[.2, .4], [.3, .6]])
    destination = np.array([[0., 1.], [1., 0.]])
    emission = np.array([[.8, .2], [.3, .7], [.6, .4]])
    transition = np.zeros((4, 4))
    for state in range(2):
        for age in range(2):
            origin = state * 2 + age
            transition[origin, state * 2 + min(age + 1, 1)] = 1 - hazard[state, age]
            transition[origin, (1 - state) * 2] = hazard[state, age]
    paths, weights = [], []
    for path in product(range(4), repeat=3):
        prob = initial[path[0]] * emission[0, path[0] // 2]
        for day in range(1, 3):
            prob *= transition[path[day - 1], path[day]] * emission[day, path[day] // 2]
        paths.append(path)
        weights.append(prob)
    weights = np.asarray(weights)
    expected = np.zeros((3, 2, 2))
    for path, prob in zip(paths, weights):
        for day, expanded in enumerate(path):
            expected[day, expanded // 2, expanded % 2] += prob / weights.sum()
    a, b, _, _, ll = model._fb(np.log(emission)[None],
        np.broadcast_to(hazard, (1, 3, 2, 2)),
        np.broadcast_to(destination, (1, 3, 2, 2)), initial.reshape(1, 2, 2))
    np.testing.assert_allclose((a * b)[0], expected)
    assert ll == pytest.approx(np.log(weights.sum()))


def test_downscaler_grid_mismatch_is_not_silently_reshaped():
    ds = observations()
    hindcast = ds.PRCP.expand_dims(member=[0, 1])
    down = DynamicalDownscaler(coupling="none", window=0).fit(hindcast, ds, [1999, 2000])
    forecast = hindcast.assign_coords(X=[20., 30.])
    with pytest.raises(ValueError, match="exactly match"):
        down.downscale(forecast, 2001)


def test_downscaler_ecc_works_with_only_seasonal_input():
    ds = observations("2001-07-01", "2001-09-30")
    hindcast = ds.PRCP.expand_dims(member=[0, 1])
    down = DynamicalDownscaler(coupling="ecc", window=7).fit(hindcast, ds, [2001])
    forecast = hindcast.assign_coords(T=season_dates(2002))
    assert down.downscale(forecast, 2002).PRCP.shape == (2, 92, 1, 2)


def test_glm_missing_user_totals_stay_missing(fitted_glm):
    out = fitted_glm.totals_to_covariate(np.array([[300., np.nan], [np.nan, 400.]]))
    assert np.isnan(out[0, 1]) and np.isnan(out[1, 0])


def test_glm_masks_invalid_forecast_triples(fitted_glm):
    p = xr.DataArray(np.array([[[.3, np.nan]], [[.4, .5]], [[.3, .5]]]),
        dims=("probability", "Y", "X"),
        coords={"probability": ["PB", "PN", "PA"], "Y": [9.], "X": [2., 3.]})
    out = fitted_glm.generate(2004, 4, probabilities=p)
    assert np.isnan(out.PRCP.sel(X=3.)).all()
    assert np.isfinite(out.PRCP.sel(X=2.)).all()


def test_glm_masks_member_with_missing_covariate(fitted_glm):
    out = fitted_glm.generate(2004, 3, covariate=np.array([[0., 0.], [np.nan, 0.], [1., 0.]]),
                             calibrated_covariate=False)
    assert np.isnan(out.PRCP.sel(member=1, X=2.)).all()
    assert np.isfinite(out.PRCP.sel(member=0, X=2.)).all()


def test_glm_calibration_ignores_missing_member_for_finite_moments(fitted_glm, monkeypatch):
    monkeypatch.setattr(fitted_glm, "transmission_", {"a": np.ones(2), "b": np.zeros(2),
        "nu": np.zeros(2), "A": np.ones(2)})
    covariate = np.array([[0., 0.], [np.nan, 0.], [1., 0.], [2., 0.]])
    out = fitted_glm.generate(2004, 4, covariate=covariate)
    assert np.isnan(out.PRCP.sel(member=1, X=2.)).all()
    assert np.isfinite(out.PRCP.sel(member=0, X=2.)).all()


def test_glm_uses_each_training_year_calendar(monkeypatch):
    import was_disaggregation.glm as glm_module
    original = glm_module.fit_probit
    designs = []
    def capture(X, *args, **kwargs):
        designs.append(X.copy())
        return original(X, *args, **kwargs)
    monkeypatch.setattr(glm_module, "fit_probit", capture)
    model = GLMWeatherGenerator(climatology=(1999, 2003), harmonics=1,
        spatial="independent", calibrate_covariate=False).fit(observations())
    expected = np.concatenate([model._harm(season_dates(y)) for y in range(1999, 2004)])
    np.testing.assert_allclose(designs[0][0, :, :3], expected)
    assert not np.array_equal(expected[:92], expected[92:184])


def test_glm_named_covariate_dimensions_are_respected(fitted_glm):
    array = xr.DataArray(np.array([[[0., 1., 2.]], [[3., 4., 5.]]]),
        dims=("X", "Y", "member"), coords={"X": [2., 3.], "Y": [9.], "member": [0, 1, 2]})
    out = fitted_glm.generate(2004, 3, covariate=array, calibrated_covariate=False)
    np.testing.assert_allclose(out.covariate_used.values, array.transpose("member", "Y", "X").values)


def test_glm_constant_continuous_variables_have_valid_fit():
    model = GLMWeatherGenerator(climatology=(1999, 2003), harmonics=0,
        spatial="independent", calibrate_covariate=False).fit(observations(constant_continuous=True))
    assert np.isfinite(model.chol_cont_).all()
    assert np.isfinite(model.generate(2004, 3).TMAX).all()


def test_glm_degenerate_calibration_and_empty_transmission_are_defined():
    ds = observations()
    ds.PRCP.values[:] = 0.
    model = GLMWeatherGenerator(climatology=(1999, 2003), harmonics=0,
        spatial="independent", calibration_members=6).fit(ds)
    assert all(np.isfinite(v).all() for v in model.transmission_.values())
    generated = model.generate(2004, 3, covariate=np.zeros((3, 2)), calibrated_covariate=False)
    report = model.covariate_transmission(generated)
    assert np.isnan(report["slope_mean"])


def test_synthetic_demonstration_does_not_spread_ocean_missingness():
    ds = observations()
    ds.PRCP.values[:, :, 1] = np.nan
    out = synthetic_model_ensemble(ds, [2001], n_members=2)
    assert np.isnan(out.sel(X=3.)).all()
    assert np.isfinite(out.sel(X=2.)).all()
