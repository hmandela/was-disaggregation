"""Regression checks for label safety, holdouts and optional generative ML."""
import builtins
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from was_disaggregation import generative as g


def rainfall(years=(2001, 2002), value=1.0):
    dates = pd.DatetimeIndex(np.concatenate([g.season_dates(y, (7,)).values for y in years]))
    return xr.DataArray(np.full((len(dates), 2, 2), value), dims=("T", "Y", "X"),
                        coords={"T": dates, "Y": [1.0, 2.0], "X": [3.0, 4.0]}, name="PRCP")


def prepared_model():
    obj = g.GenerativeDownscaler(months=(7,), factor=2)
    obj.y_, obj.x_ = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    obj.net_ = object()  # downscale's numpy pipeline can be checked without PyTorch.
    obj.scale_, obj.extra_names_ = 1.0, []
    obj.valid_mask_ = np.array([[True, True], [True, False]])
    return obj


def test_optional_torch_error_and_numpy_helpers(monkeypatch):
    original = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("PyTorch deliberately absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    obj = g.GenerativeDownscaler()
    assert obj.factor == 2
    assert g.coarse_block_mean(np.ones((2, 2)), 2).item() == 1
    with pytest.raises(ImportError, match=r"was-disaggregation\[ml\]"):
        g._torch()


@pytest.mark.parametrize("params", [
    {"factor": 0}, {"factor": 1.5}, {"time_window": -1}, {"alpha": 1.1},
    {"months": (7, 9)}, {"batch_size": 0}, {"epochs": 0},
    {"method": "diffusion", "n_steps": 1}, {"patch_size": 1}, {"pool_scales": (2,)},
])
def test_bad_constructor_parameters(params):
    with pytest.raises(ValueError):
        g.GenerativeDownscaler(**params)


def test_coarse_blocks_include_partial_edges_and_missing_cells():
    a = np.array([[1.0, 3.0, 5.0], [np.nan, 5.0, 7.0], [9.0, 11.0, np.nan]])
    result = g.coarse_block_mean(a, 2)
    np.testing.assert_allclose(result, [[3.0, 6.0], [10.0, np.nan]], equal_nan=True)
    with pytest.raises(ValueError, match="factor"):
        g.coarse_block_mean(a, 0)


def test_signed_ar1_noise_and_run_restart():
    shape, rho, runs = (6, 2), -0.75, np.array([0, 0, 0, 1, 1, 1])
    expected = np.random.default_rng(3).standard_normal(shape).astype(np.float32)
    for t in range(1, len(runs)):
        if runs[t] == runs[t - 1]:
            expected[t] = rho * expected[t - 1] + np.sqrt(1 - rho**2) * expected[t]
    np.testing.assert_allclose(g.ar1_noise(np.random.default_rng(3), shape, rho, runs), expected, atol=1e-7)
    for invalid in (1, -1, np.nan):
        with pytest.raises(ValueError, match="rho"):
            g.ar1_noise(np.random.default_rng(3), shape, invalid)


def test_training_never_falls_back_to_heldout_year():
    obj = g.GenerativeDownscaler(months=(7,))
    with pytest.raises(ValueError, match="at least two seasons"):
        obj.fit(rainfall((2001,)), [2001])
    with pytest.raises(ValueError, match="no training seasons remain"):
        obj.fit(rainfall(), [2001, 2002], validation_years=[2001, 2002])


def test_training_defaults_to_latest_year_and_checks_predictor_grid(monkeypatch):
    calls = []
    obj = g.GenerativeDownscaler(months=(7,))
    original = obj._season_data

    def record(prcp, years, predictors=None):
        calls.append(years)
        return original(prcp, years, predictors)

    monkeypatch.setattr(obj, "_season_data", record)
    monkeypatch.setattr(g, "_torch", lambda: (_ for _ in ()).throw(ImportError("no torch")))
    with pytest.raises(ImportError, match="no torch"):
        obj.fit(rainfall(), [2002, 2001])
    assert calls == [[2001], [2002]]
    wrong_grid = rainfall().assign_coords(Y=[2.0, 1.0])
    with pytest.raises(ValueError, match="observation grid"):
        obj.fit(rainfall(), [2001, 2002], predictors={"TCWV": wrong_grid})


def test_static_grid_labels_must_match():
    obj = prepared_model()
    bad = xr.DataArray(np.ones((2, 2)), dims=("Y", "X"), coords={"Y": [2., 1.], "X": [3., 4.]})
    with pytest.raises(ValueError, match="static predictor"):
        obj._static({"elevation": bad})


def test_sampling_lags_stay_inside_members_and_missing_domain_is_masked(monkeypatch):
    obj = prepared_model()
    model = rainfall((2001,)).isel(T=[0, 1, 3]).expand_dims(member=[11, 22])

    def fake_sample(feats, n_seq, T, n_samples, rng, noise_rho, batch_days):
        np.testing.assert_array_equal(feats["lag"],
                                      [[0, 0, 1], [0, 1, 1], [2, 2, 2],
                                       [3, 3, 4], [3, 4, 4], [5, 5, 5]])
        return np.ones((n_seq * n_samples, T, 2, 2), dtype=np.float32)

    monkeypatch.setattr(obj, "_sample_sequences", fake_sample)
    result = obj.downscale(model, n_samples=2)
    assert result.PRCP.dims == ("member", "T", "Y", "X")
    assert result.sizes["member"] == 4
    assert np.isnan(result.PRCP.values[..., 1, 1]).all()
    assert np.isfinite(result.PRCP.values[..., 0, 0]).all()


@pytest.mark.parametrize("params", [{"n_samples": 0}, {"batch_days": 0}, {"noise_rho": 1}, {"noise_rho": np.nan}])
def test_sampling_parameters(params):
    with pytest.raises(ValueError):
        prepared_model().downscale(rainfall(), **params)


@pytest.mark.parametrize("mismatch", ["T", "Y", "member"])
def test_sampling_rejects_misaligned_predictors(mismatch):
    obj = prepared_model()
    obj.extra_names_ = ["TCWV"]
    obj.extra_mean_, obj.extra_sd_ = np.array([0.0]), np.array([1.0])
    model = rainfall().expand_dims(member=[11, 22])
    extra = model.copy()
    if mismatch == "T":
        extra = extra.assign_coords(T=extra["T"].values + np.timedelta64(1, "D"))
    else:
        extra = extra.isel({mismatch: slice(None, None, -1)})
    with pytest.raises(ValueError, match="coordinates|grid"):
        obj.downscale(model, predictors={"TCWV": extra})


def test_perfect_prognosis_uses_observed_predictors_without_model_correction(monkeypatch):
    obj = prepared_model()
    captured = {}

    def capture(model, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(obj, "downscale", capture)
    result = obj.sample_perfect_prognosis(rainfall(), [2002], predictors={"TCWV": rainfall()})
    assert result.sizes["T"] == 31
    assert captured["correct_predictors"] is False
    np.testing.assert_array_equal(captured["predictors"]["TCWV"]["T"], result["T"])


@pytest.mark.parametrize("method", ["crps", "cgan", "diffusion", "flow"])
def test_torch_fit_sample_and_checkpoint_roundtrip(method, tmp_path):
    torch = pytest.importorskip("torch", reason="Optional PyTorch is not installed")
    torch.set_num_threads(1)
    obj = g.GenerativeDownscaler(method=method, months=(7,), factor=2, width=4, depth=1,
                                ensemble_size=2, epochs=1, batch_size=64, n_critic=1,
                                n_steps=2, device="cpu", noise_channels=1, noise_dim=2)
    observations = rainfall(value=0.0)
    obj.fit(observations, [2001, 2002])
    assert np.isfinite(obj.history_.val_loss).all()
    if method in ("diffusion", "flow"):
        assert obj.sigma_data_ > 0
    sample = obj.downscale(observations.isel(T=slice(0, 2)), seed=4)
    assert sample.PRCP.shape == (1, 2, 2, 2)
    assert np.isfinite(sample.PRCP.values).all()
    path = tmp_path / "generator.pt"
    obj.save(path)
    restored = g.GenerativeDownscaler.load(path, device="cpu")
    xr.testing.assert_identical(sample, restored.downscale(observations.isel(T=slice(0, 2)), seed=4))


def test_torch_crps_alpha_endpoints_match_exact_scores():
    torch = pytest.importorskip("torch", reason="Optional PyTorch is not installed")
    ensemble = torch.tensor([[[[0.0]], [[2.0]]]])
    target, mask = torch.ones((1, 1, 1)), torch.ones((1, 1, 1))
    assert g._afcrps(torch, ensemble, target, mask, alpha=0).item() == pytest.approx(0.5)
    assert g._afcrps(torch, ensemble, target, mask, alpha=1).item() == pytest.approx(0.0)
    assert g._energy_score(torch, ensemble, target, mask).item() == pytest.approx(0.0)


@pytest.mark.parametrize("method", ["diffusion", "flow"])
def test_torch_residual_training_and_checkpoint_roundtrip(method, tmp_path):
    torch = pytest.importorskip("torch", reason="Optional PyTorch is not installed")
    torch.set_num_threads(1)
    obj = g.GenerativeDownscaler(method=method, residual=True, months=(7,), factor=2,
                                width=4, depth=1, epochs=1, batch_size=64,
                                n_steps=2, device="cpu")
    observations = rainfall()
    observations.values *= 1 + np.arange(observations.sizes["T"])[:, None, None] % 5
    obj.fit(observations, [2001, 2002])
    assert list(obj.history_.stage) == ["mean", "main"]
    assert np.isfinite(obj.history_.val_loss).all()
    assert obj.sigma_data_ > 0
    sample = obj.downscale(observations.isel(T=slice(0, 2)), seed=4)
    assert np.isfinite(sample.PRCP).all()
    path = tmp_path / "residual_generator.pt"
    obj.save(path)
    restored = g.GenerativeDownscaler.load(path, device="cpu")
    xr.testing.assert_identical(sample, restored.downscale(observations.isel(T=slice(0, 2)), seed=4))
