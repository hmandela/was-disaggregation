"""Public CLI and reproducibility across actual Dask tile/member boundaries."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from was_disaggregation import WeatherGenerator, generate_dask, season_dates
from was_disaggregation.cli import _constraints, main


def sample():
    dates = pd.DatetimeIndex(np.concatenate([season_dates(y, (7,)).values for y in range(2001, 2009)]))
    rng = np.random.default_rng(99)
    rain = rng.gamma(2, 3, size=(len(dates), 2, 2))
    rain[rng.random(rain.shape) < .45] = 0
    rain *= np.repeat(np.arange(1, 9), 31)[:, None, None]
    obs = xr.Dataset({"PRCP": (("T", "Y", "X"), rain)},
                     coords={"T": dates, "Y": [6., 7.], "X": [1., 2.]})
    obs.PRCP.attrs["units"] = "mm d-1"
    prob = xr.DataArray(np.full((3, 2, 2), 1/3), dims=("probability", "Y", "X"),
                        coords={"probability": ["PB", "PN", "PA"], "Y": obs.Y, "X": obs.X})
    return obs, prob


@pytest.mark.parametrize("conditioning", ["mean", "mixture"])
def test_tiling_and_batching_preserve_independent_samples(conditioning):
    obs, prob = sample()
    kwargs = dict(months=(7,), climatology=(2001, 2008), spatial="independent", seed=5,
                  conditioning=conditioning)
    full = WeatherGenerator(**kwargs).fit(obs, prob).generate(2009, n_members=7)
    tiled = generate_dask(obs, prob, 2009, n_members=7, tile_shape=(1, 1), member_batch=3, **kwargs)
    assert hasattr(tiled.PRCP.data, "compute")
    xr.testing.assert_allclose(tiled.compute(scheduler="synchronous"), full)
    other = generate_dask(obs, prob, 2009, n_members=7, tile_shape=(2, 2), member_batch=5, **kwargs)
    xr.testing.assert_allclose(tiled.compute(scheduler="synchronous"), other.compute(scheduler="synchronous"))


def test_distance_tiling_reuses_one_kernel_for_same_realizations():
    obs, prob = sample()
    kwargs = dict(months=(7,), climatology=(2001, 2008), spatial="distance", seed=8, n_features=12)
    fitted = WeatherGenerator(**kwargs).fit(obs, prob)
    kernels = fitted.spatial_models_
    a = generate_dask(obs, prob, 2009, n_members=3, tile_shape=(1, 1), member_batch=2,
                      spatial_models=kernels, **kwargs).compute(scheduler="synchronous")
    b = generate_dask(obs, prob, 2009, n_members=3, tile_shape=(2, 2), member_batch=3,
                      spatial_models=kernels, **kwargs).compute(scheduler="synchronous")
    xr.testing.assert_allclose(a, b)


def test_total_window_cli_is_applied_without_other_constraints():
    args = SimpleNamespace(total_window=["07-01", "09-30"], onset=None, dry_spell=None,
                           cessation=None, tolerance=.01, onset_search=["05-01", "09-30"])
    constraints = _constraints(args)
    assert len(constraints) == 1
    assert constraints[0].probabilities is None


def test_cli_reads_labelled_constraint_among_other_variables(tmp_path):
    _, prob = sample()
    prob = prob.assign_coords(probability=["early", "normal", "late"])
    path = tmp_path / "onset.nc"
    xr.Dataset({"unrelated": (("T", "Y", "X"), np.zeros((2, 2, 2))),
                "forecast": prob}).to_netcdf(path)
    args = SimpleNamespace(total_window=None, onset=str(path), dry_spell=None,
                           cessation=None, tolerance=.01, onset_search=["05-01", "09-30"])
    result = _constraints(args)
    assert list(result[0].probabilities.probability.values) == ["PB", "PN", "PA"]


def test_cli_version_and_invalid_integer_options(capsys):
    with pytest.raises(SystemExit) as done:
        main(["--version"])
    assert done.value.code == 0
    assert "0.8.0" in capsys.readouterr().out
    obs, prob = sample()
    with pytest.raises(ValueError, match="n_members must be an integer"):
        generate_dask(obs, prob, 2009, n_members=1.5)


def test_cli_netcdf_generation_preserves_integer_class_labels(tmp_path):
    obs, prob = sample()
    obs_path, prob_path, output = (tmp_path / name for name in ("obs.nc", "forecast.nc", "generated.nc"))
    obs.to_netcdf(obs_path)
    prob.to_dataset(name="forecast").to_netcdf(prob_path)
    main(["--obs", str(obs_path), "--forecast", str(prob_path), "--output", str(output),
          "--year", "2009", "--months", "7", "--climatology", "2001", "2008",
          "--members", "2", "--member-batch", "1", "--tile", "1", "1",
          "--spatial", "independent", "--conditioning", "mixture"])
    with xr.open_dataset(output) as result:
        assert result.PRCP.shape == (2, 31, 2, 2)
        assert result.tercile_class.dtype == np.int8
        assert bool(np.isfinite(result.PRCP).all())
