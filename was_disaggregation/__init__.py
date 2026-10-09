"""Forecast-conditioned stochastic weather generation on xarray grids.

This module exports the public API; it is package infrastructure.
Scientific authors, article references and variant limits appear in the
corresponding implementation modules, separate from software authorship.
"""
from ._version import __version__
from .data import open_observations, load_probabilities, seasonal_cube, season_dates
from .model import WeatherGenerator
from .rainfall import fit_wilks_forecast_plane
from .srg3 import SRG3Fit, fit_srg3, simulate_srg3
from .scalable import generate_dask
from .nonparametric import ForecastAnalogGenerator, ForecastSchaakeGenerator, schaake_shuffle, enso_rank_weights
from .attributes import SeasonalTotal, OnsetDate, CessationDate, MaxDrySpell
from .mre import SeasonalConstraint, constrained_year_weights
from .diagnostics import attribute_diagnostics, select_members
from .glm import GLMWeatherGenerator, CoefficientGP
from .bayesian import BayesianGLMWeatherGenerator
from .apipattanavis import ApipattanavisGenerator
from .spatial import DenseGaussianSpatialField, GaussianSpatialField
from .dynamical import (DailyBiasCorrector, ExternalCorrector, NHMM, NHSMM, state_durations, nhmm_predictors, ensemble_copula_coupling,
                        preferential_dates, minimum_divergence_selection, minimum_divergence_dates,
                        DynamicalDownscaler, synthetic_model_ensemble)
from .validation import (HindcastExperiment, model_tercile_probabilities, ensemble_normal_scores, rank_histogram,
                         reliability_table, daily_scores, crps_ensemble, rps_ensemble,
                         brier_score, brier_decomposition)
from .generative import GenerativeDownscaler                      # needs PyTorch only when fitted
__all__ = ["WeatherGenerator", "fit_wilks_forecast_plane", "SRG3Fit", "fit_srg3", "simulate_srg3",
           "ForecastAnalogGenerator", "ForecastSchaakeGenerator", "generate_dask",
           "open_observations", "load_probabilities", "seasonal_cube", "season_dates", "schaake_shuffle",
           "enso_rank_weights", "SeasonalTotal", "OnsetDate", "CessationDate", "MaxDrySpell",
           "SeasonalConstraint", "constrained_year_weights", "attribute_diagnostics", "select_members",
           "GLMWeatherGenerator", "CoefficientGP", "BayesianGLMWeatherGenerator", "ApipattanavisGenerator",
           "DenseGaussianSpatialField", "GaussianSpatialField", "DailyBiasCorrector", "ExternalCorrector", "NHMM", "NHSMM", "state_durations", "nhmm_predictors",
           "ensemble_copula_coupling", "preferential_dates", "minimum_divergence_selection", "minimum_divergence_dates",
           "DynamicalDownscaler", "synthetic_model_ensemble",
           "HindcastExperiment", "model_tercile_probabilities", "ensemble_normal_scores",
           "rank_histogram", "reliability_table", "daily_scores", "crps_ensemble", "rps_ensemble",
           "brier_score", "brier_decomposition", "GenerativeDownscaler"]
