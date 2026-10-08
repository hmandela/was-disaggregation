"""Forecast-conditioned stochastic weather generation on xarray grids."""
from ._version import __version__
from .data import open_observations, load_probabilities, seasonal_cube, season_dates
from .model import WeatherGenerator
from .scalable import generate_dask
from .nonparametric import ForecastAnalogGenerator, ForecastSchaakeGenerator, schaake_shuffle, enso_rank_weights
from .attributes import SeasonalTotal, OnsetDate, CessationDate, MaxDrySpell
from .mre import SeasonalConstraint, constrained_year_weights
from .diagnostics import attribute_diagnostics, select_members
from .glm import GLMWeatherGenerator, CoefficientGP
from .dynamical import (DailyBiasCorrector, ExternalCorrector, NHMM, NHSMM, state_durations, nhmm_predictors, ensemble_copula_coupling,
                        preferential_dates, DynamicalDownscaler, synthetic_model_ensemble)
from .validation import (HindcastExperiment, model_tercile_probabilities, ensemble_normal_scores, rank_histogram,
                         reliability_table, daily_scores)
from .generative import GenerativeDownscaler                      # needs PyTorch only when fitted
__all__ = ["WeatherGenerator", "ForecastAnalogGenerator", "ForecastSchaakeGenerator", "generate_dask",
           "open_observations", "load_probabilities", "seasonal_cube", "season_dates", "schaake_shuffle",
           "enso_rank_weights", "SeasonalTotal", "OnsetDate", "CessationDate", "MaxDrySpell",
           "SeasonalConstraint", "constrained_year_weights", "attribute_diagnostics", "select_members",
           "GLMWeatherGenerator", "CoefficientGP", "DailyBiasCorrector", "ExternalCorrector", "NHMM", "NHSMM", "state_durations", "nhmm_predictors",
           "ensemble_copula_coupling", "preferential_dates", "DynamicalDownscaler", "synthetic_model_ensemble",
           "HindcastExperiment", "model_tercile_probabilities", "ensemble_normal_scores",
           "rank_histogram", "reliability_table", "daily_scores", "GenerativeDownscaler"]
