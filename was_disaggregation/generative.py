"""Conditional generative kernels for daily precipitation downscaling.

Four generators share a package convolutional backbone and data pipeline.
The mathematical objectives and sampling kernels below implement components
of the cited methods. Their architectures, trained weights, datasets and
published skill results are not reproduced by this module.

* ``method="crps"`` uses the scoring-rule minimization principle of Lorenzo
  Pacchiardi, Rilwan A. Adewoyin, Peter Dueben and Ritabrata Dutta (2024).
  Pixelwise and pooled-field CRPS, whole-field energy score and a local
  variogram score are combined. The fair CRPS/energy endpoints use independent
  ensemble-pair corrections related to Christopher A. T. Ferro (2014).
  Multi-scale pooling and the optional fair local variogram estimator are
  package extensions. ``alpha=1`` selects fair CRPS; ``alpha=0`` selects the
  empirical ensemble CRPS, with intermediate values blending the two.
* ``method="cgan"`` combines the Wasserstein gradient penalty of Ishaan
  Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin and Aaron Courville
  (2017) with the ensemble-mean content-loss mechanism of Lucy Harris,
  Andrew T. T. McRae, Matthew Chantry, Peter D. Dueben and Tim N. Palmer (2022).
  The optional variance-corrected content estimator is a package extension.
* ``method="diffusion"`` uses EDM preconditioning, noise-weighted denoising
  and Heun/churn sampling from Tero Karras, Miika Aittala, Timo Aila and Samuli
  Laine (2022). ``residual=True`` applies the regression-plus-residual
  construction of Morteza Mardani and coauthors (CorrDiff: 2023 preprint,
  2025 publication). This is a daily precipitation adaptation, without the
  original multivariate Taiwan architecture or experimental reproduction.
* ``method="flow"`` uses the Gaussian straight-path conditional flow-matching
  objective of Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel
  and Matt Le (ICLR 2023; preprint 2022). The same residual option is a package
  combination. Uniform training time is the default; logit-normal time is an
  explicit alternative objective weighting.

Default training is perfect prognosis: fine observations are targets, and
observed block means, temporal context, dates, coordinates and optional extra
fields are conditioning inputs. At forecast time, bias-corrected model-member
fields provide coarse conditioning. A seasonal member's daily trajectory is
not a deterministic forecast/observation match; actual valid-time matching
must be justified before using ``fit(..., conditioning=forecast)`` for paired
training. The default symmetric observed context is a downscaling input,
not Pacchiardi and coauthors' past-only prequential forecasting experiment.

Implemented consistency measures include matched train/sample output support,
explicit score units, independent validation years, missing-value indicators,
optional exact coarse-block means and member/sample provenance. These address
specific support, estimator or validation limitations, but do not guarantee
calibration, reliable extremes or improved seasonal skill. Correlated AR(1)
noise is a package temporal-coherence heuristic, not a learned temporal model.
See ``docs/MATH_GENERATIVE_FR.md`` for equations and remaining assumptions.

References
----------
Harris, Lucy; McRae, Andrew T. T.; Chantry, Matthew; Dueben, Peter D.; Palmer,
    Tim N. (2022). A generative deep learning approach to stochastic downscaling
    of precipitation forecasts. Journal of Advances in Modeling Earth Systems,
    14, e2022MS003120. https://doi.org/10.1029/2022MS003120
Gulrajani, Ishaan; Ahmed, Faruk; Arjovsky, Martin; Dumoulin, Vincent; Courville,
    Aaron (2017). Improved training of Wasserstein GANs. NeurIPS 2017.
    https://arxiv.org/abs/1704.00028
Karras, Tero; Aittala, Miika; Aila, Timo; Laine, Samuli (2022). Elucidating the
    design space of diffusion-based generative models. NeurIPS 2022.
    https://arxiv.org/abs/2206.00364
Mardani, Morteza; Brenowitz, Noah; Cohen, Yair; Pathak, Jaideep; Chen, Chieh-Yu;
    Liu, Cheng-Chin; Vahdat, Arash; Nabian, Mohammad Amin; Ge, Tao; Subramaniam,
    Akshay; Kashinath, Karthik; Kautz, Jan; Pritchard, Mike (2025). Residual
    corrective diffusion modeling for km-scale atmospheric downscaling.
    Communications Earth & Environment, 6, 124.
    https://doi.org/10.1038/s43247-025-02042-5
    Original 2023 preprint: https://arxiv.org/abs/2309.15214
Lipman, Yaron; Chen, Ricky T. Q.; Ben-Hamu, Heli; Nickel, Maximilian; Le,
    Matt (2023). Flow matching for generative modeling. ICLR 2023.
    https://arxiv.org/abs/2210.02747 (first submitted 2022).
Pacchiardi, Lorenzo; Adewoyin, Rilwan A.; Dueben, Peter; Dutta, Ritabrata
    (2024). Probabilistic forecasting with generative networks via scoring
    rule minimization. Journal of Machine Learning Research, 25(45), 1-64.
    https://jmlr.org/papers/v25/23-0038.html
Ferro, Christopher A. T. (2014). Fair scores for ensemble forecasts.
    Quarterly Journal of the Royal Meteorological Society, 140, 1917-1923.
    https://doi.org/10.1002/qj.2270
Scheuerer, Michael; Hamill, Thomas M. (2015). Variogram-based proper scoring
    rules for probabilistic forecasts of multivariate quantities.
    Monthly Weather Review, 143, 1321-1334.
    https://doi.org/10.1175/MWR-D-14-00269.1

PyTorch is optional: ``pip install 'was-disaggregation[ml]'``.
"""
from __future__ import annotations

import copy
import math
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .data import season_dates, validate_months

__all__ = ["GenerativeDownscaler", "coarse_block_mean", "ar1_noise"]

_METHODS = ("crps", "cgan", "diffusion", "flow")


def _torch():
    try:
        import torch
    except ImportError as err:                                                  # pragma: no cover
        raise ImportError("was_disaggregation.generative needs PyTorch: pip install torch "
                          "(or pip install 'was-disaggregation[ml]')") from err
    return torch


# ---------------------------------------------------------------------------
# numpy helpers
# ---------------------------------------------------------------------------
def _integer(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _years(values, name="years"):
    values = [_integer(name, y) for y in values]
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be nonempty and contain unique years")
    return sorted(values)


def _daily_dates(values):
    try:
        dates = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as err:
        raise ValueError("T must contain Gregorian daily datetime coordinates") from err
    if (dates.empty or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
            or not dates.equals(dates.normalize())):
        raise ValueError("T must contain nonempty, unique, increasing daily dates at midnight")
    return dates


def coarse_block_mean(fine, factor):
    """Block means of (..., H, W) over factor x factor blocks (NaN-aware, edge
    blocks padded) -> (..., ceil(H/f), ceil(W/f))."""
    fine = np.asarray(fine, float)
    f = _integer("factor", factor)
    if fine.ndim < 2 or any(n == 0 for n in fine.shape[-2:]):
        raise ValueError("fine must have two nonempty spatial dimensions")
    if f == 1:
        return fine.copy()
    H, W = fine.shape[-2:]
    Hp, Wp = -(-H // f) * f, -(-W // f) * f
    pad = [(0, 0)] * (fine.ndim - 2) + [(0, Hp - H), (0, Wp - W)]
    x = np.pad(fine, pad, constant_values=np.nan)
    x = x.reshape(*x.shape[:-2], Hp // f, f, Wp // f, f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=(-3, -1))


def ar1_noise(rng, shape, rho, runs=None):
    """Standard-normal noise (T, ...) with lag-1 correlation ``rho`` along the
    first axis, restarted at each new contiguous run (``runs``: (T,) run ids)."""
    if not np.isfinite(rho) or not -1 < rho < 1:
        raise ValueError("rho must be finite and strictly between -1 and 1")
    if not shape or shape[0] < 0:
        raise ValueError("shape must have a time dimension")
    if runs is not None and np.asarray(runs).shape != (shape[0],):
        raise ValueError("runs must have one id per time step")
    e = rng.standard_normal(shape).astype(np.float32)
    if rho == 0:
        return e
    s = math.sqrt(1 - rho ** 2)
    for t in range(1, shape[0]):
        if runs is None or runs[t] == runs[t - 1]:
            e[t] = rho * e[t - 1] + s * e[t]
    return e


def _runs(dates):
    """Contiguous-day run ids of a DatetimeIndex."""
    d = pd.DatetimeIndex(dates)
    if d.empty:
        return np.empty(0, dtype=int)
    gap = np.r_[True, np.diff(d.values) != np.timedelta64(1, "D")]
    return np.cumsum(gap) - 1


def _lag_index(runs, k, context="symmetric"):
    """Context indices, clamped inside each contiguous run."""
    T = len(runs)
    offsets = np.arange(-k, k + 1) if context == "symmetric" else np.arange(-k, 1)
    idx = np.arange(T)[:, None] + offsets[None]
    start = np.zeros(T, int)
    end = np.zeros(T, int)
    for r in np.unique(runs):
        w = np.flatnonzero(runs == r)
        start[w], end[w] = w[0], w[-1]
    return np.clip(idx, start[:, None], end[:, None])


# ---------------------------------------------------------------------------
# networks (built lazily so that importing the package does not need torch)
# ---------------------------------------------------------------------------
_NETS = {}


def _nets():
    if _NETS:
        return _NETS
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional

    class FourierEmbedding(nn.Module):
        def __init__(self, n=16, scale=16.0, out=64):
            super().__init__()
            self.register_buffer("freqs", torch.randn(n) * scale)
            self.mlp = nn.Sequential(nn.Linear(2 * n, out), nn.SiLU(), nn.Linear(out, out))

        def forward(self, c):                                                   # c (B,)
            a = 2 * math.pi * c[:, None] * self.freqs[None]
            return self.mlp(torch.cat([a.cos(), a.sin()], -1))

    class ResBlock(nn.Module):
        def __init__(self, w, dilation, cond_dim):
            super().__init__()
            self.c1 = nn.Conv2d(w, w, 3, padding=dilation, dilation=dilation)
            self.c2 = nn.Conv2d(w, w, 3, padding=1)
            nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)
            self.film = nn.Linear(cond_dim, 2 * w) if cond_dim else None

        def forward(self, h, cond=None):
            r = self.c1(F.silu(h))
            if self.film is not None:
                s, b = self.film(cond).chunk(2, -1)
                r = r * (1 + s[..., None, None]) + b[..., None, None]
            return h + self.c2(F.silu(r))

    class Backbone(nn.Module):
        """Fully convolutional residual net with dilated 3x3 convolutions (any grid size)."""

        def __init__(self, in_ch, out_ch=1, width=48, depth=6, cond_dim=0, max_dilation=8):
            super().__init__()
            dil = [min(2 ** (i % 4), max_dilation) for i in range(depth)]
            self.inp = nn.Conv2d(in_ch, width, 3, padding=1)
            self.blocks = nn.ModuleList([ResBlock(width, d, cond_dim) for d in dil])
            self.out = nn.Conv2d(width, out_ch, 3, padding=1)

        def forward(self, x, cond=None):
            h = self.inp(x)
            for b in self.blocks:
                h = b(h, cond)
            return self.out(F.silu(h))

    class Generator(nn.Module):
        """x (B,C,H,W) + spatial noise (B,k,H,W) + global noise (B,g) -> (B,1,H,W)."""

        def __init__(self, in_ch, noise_channels, noise_dim, width, depth, max_dilation):
            super().__init__()
            self.k, self.g = noise_channels, noise_dim
            self.embed = nn.Sequential(nn.Linear(noise_dim, 64), nn.SiLU(), nn.Linear(64, 64)) if noise_dim else None
            self.net = Backbone(in_ch + noise_channels, 1, width, depth, 64 if noise_dim else 0, max_dilation)

        def forward(self, x, zs, zg):
            cond = self.embed(zg) if self.embed is not None else None
            return self.net(torch.cat([x, zs], 1) if self.k else x, cond)

    class Critic(nn.Module):
        """WGAN critic on (conditioning, field): convolutions, global mean, linear."""

        def __init__(self, in_ch, width, depth, max_dilation):
            super().__init__()
            dil = [min(2 ** (i % 4), max_dilation) for i in range(depth)]
            layers = [nn.Conv2d(in_ch + 1, width, 3, padding=1)]
            for d in dil:
                layers += [nn.LeakyReLU(0.2), nn.Conv2d(width, width, 3, padding=d, dilation=d)]
            self.conv = nn.Sequential(*layers)
            self.head = nn.Sequential(nn.LeakyReLU(0.2), nn.Linear(2 * width, 1))

        def forward(self, x, y):
            h = self.conv(torch.cat([x, y], 1))
            return self.head(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))

    class Denoiser(nn.Module):
        """F(noisy field, conditioning, noise level) for EDM diffusion / flow matching."""

        def __init__(self, in_ch, width, depth, max_dilation):
            super().__init__()
            self.embed = FourierEmbedding(out=64)
            self.net = Backbone(in_ch + 1, 1, width, depth, 64, max_dilation)

        def forward(self, y, x, c):
            return self.net(torch.cat([y, x], 1), self.embed(c))

    _NETS.update(Backbone=Backbone, Generator=Generator, Critic=Critic, Denoiser=Denoiser)
    return _NETS


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def _masked_pool(torch, y, m, s):
    if s == 1:
        return y, m
    F = torch.nn.functional
    num = F.avg_pool2d(y * m, s, ceil_mode=True)
    den = F.avg_pool2d(m, s, ceil_mode=True)
    return num / den.clamp_min(1e-6), (den > 0).float()


def _afcrps(torch, ens, y, m, alpha):
    """Blend empirical and fair CRPS, with Ferro (2014)'s fair endpoint.

    ``alpha=1`` uses off-diagonal M(M-1) pair normalization; ``alpha=0``
    uses empirical M**2 normalization. Requires independent ensemble draws
    for the fair interpretation. Reference: https://doi.org/10.1002/qj.2270.
    Shapes: ens (B,M,...), y (B,...), mask m (B,...).
    """
    M = ens.shape[1]
    if M < 2:
        raise ValueError("fair CRPS requires at least two independent ensemble draws")
    t1 = (ens - y[:, None]).abs().mean(1)
    t2 = (ens[:, :, None] - ens[:, None]).abs().sum((1, 2)) * (M - 1 + alpha) / (2 * M * M * (M - 1))
    return ((t1 - t2) * m).sum() / m.sum().clamp_min(1.0)


def _energy_score(torch, ens, y, m):
    """Fair energy score for scoring-rule training after Pacchiardi, Adewoyin,
    Dueben and Dutta (2024), https://jmlr.org/papers/v25/23-0038.html.

    Off-diagonal pairs remove independent finite-ensemble estimator bias;
    sqrt(valid-cell-count) normalization is a package scaling convention.
    This implements a score kernel, not the authors' forecasting experiment.
    """
    B, M = ens.shape[:2]
    if M < 2:
        raise ValueError("fair energy score requires at least two independent ensemble draws")
    e = (ens * m[:, None]).flatten(2)
    t = (y * m).flatten(1)
    n = m.flatten(1).sum(1).clamp_min(1.0).sqrt()
    t1 = (e - t[:, None]).norm(dim=-1).mean(1)
    d = (e[:, :, None] - e[:, None]).norm(dim=-1)
    t2 = d.sum((1, 2)) / (2 * M * (M - 1))
    valid = m.flatten(1).sum(1) > 0
    return (((t1 - t2) / n) * valid).sum() / valid.sum().clamp_min(1)


_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0))


def _variogram_score(torch, ens, y, m, p=0.5, offsets=_OFFSETS, fair=True):
    """Local variogram score after Michael Scheuerer and Thomas M. Hamill
    (2015), https://doi.org/10.1175/MWR-D-14-00269.1.

    Local offsets approximate the full component-pair score. The package's
    ``fair`` extension subtracts Var(member differences)/M, giving an unbiased
    population-score estimator for independent draws; finite values may be
    negative. Minimizing this score does not guarantee correct covariability.
    Shapes: ens (B,M,H,W), y (B,H,W), m (B,H,W).
    """
    if fair and ens.shape[1] < 2:
        raise ValueError("fair variogram score requires at least two ensemble draws")
    tot, n = 0.0, 0.0
    H, W = y.shape[-2:]
    for dy, dx in offsets:
        if dy >= H or abs(dx) >= W:
            continue
        ys = slice(0, H - dy)
        xa, xb = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
        y1, y2 = y[..., ys, xa], y[..., dy:, xb]
        e1, e2 = ens[..., ys, xa], ens[..., dy:, xb]
        mm = m[..., ys, xa] * m[..., dy:, xb]
        # Use an explicit zero-gradient numerical convention: p<1 otherwise
        # gives infinite derivatives at exactly equal (notably dry) cells.
        od, ed = (y1 - y2).abs(), (e1 - e2).abs()
        vo = torch.where(od > 0, od.clamp_min(1e-12) ** p, torch.zeros_like(od))
        members = torch.where(ed > 0, ed.clamp_min(1e-12) ** p, torch.zeros_like(ed))
        ve = members.mean(1)
        term = (vo - ve) ** 2
        if fair:
            term = term - members.var(1, unbiased=True) / ens.shape[1]
        tot = tot + (term * mm).sum()
        n = n + mm.sum()
    return tot / max(float(n), 1.0)


# ---------------------------------------------------------------------------
# the downscaler
# ---------------------------------------------------------------------------
class GenerativeDownscaler:
    """Stochastic daily precipitation downscaling with a conditional network.

    Perfect prognosis is the default; ``fit(conditioning=...)`` permits
    justified valid-time paired training. Scientific kernels: Harris, McRae,
    Chantry, Dueben and Palmer (2022), cGAN content; Gulrajani, Ahmed, Arjovsky,
    Dumoulin and Courville (2017), WGAN-GP; Karras, Aittala, Aila and Laine
    (2022), EDM; Mardani and coauthors (2023/2025), residual correction;
    Lipman, Chen, Ben-Hamu, Nickel and Le (2023), conditional flow matching;
    Pacchiardi, Adewoyin, Dueben and Dutta (2024), scoring-rule training.
    Full bibliographic references appear in the module docstring.

    The common dilated backbone and daily univariate precipitation pipeline
    are package variants. Numerical consistency checks and optional fair
    scores do not demonstrate superior calibration or reproduce the authors'
    original networks, datasets, experiments or published performance.

    Parameters
    ----------
    method : "crps" | "cgan" | "diffusion" | "flow"
    factor : coarse block size in fine cells (e.g. 10 for 1 deg C3S -> 0.1 deg AgERA5)
    months : season months used for training (and the default for ``downscale``)
    time_window : k, the coarse field of days t-k..t+k is given to the network
    context : "symmetric" or "past" (t-k..t). Future forecast fields may be
        known at issue time; future *observations* are not causal predictors.
    width, depth : backbone channels and number of dilated residual blocks
    noise_channels, noise_dim : spatial noise maps and global noise vector (crps, cgan)
    ensemble_size : members drawn per training sample (crps loss and cgan content loss, default 8)
    alpha, energy_weight, variogram_weight, pool_scales : afCRPS alpha, weights of the
        energy score and of the local variogram score (p = 0.5), and the pooling
        sizes of the multi-scale CRPS (crps)
    content_weight, n_critic, gp_weight : Harris and coauthors (2022) content
        loss, critic steps per generator step and Gulrajani and coauthors
        (2017) gradient penalty (cgan)
    residual : regression plus residual after Mardani and coauthors
        (2023/2025) for diffusion; its use with flow is a package combination
    n_steps : sampler steps (diffusion, flow)
    epochs, batch_size, lr, patience, max_minutes : training budget (Adam, cosine
        learning-rate decay to lr/20); the state with the lowest validation loss is kept.
        Defaults per method: crps 40 epochs, lr 1e-3; cgan 30, 2e-4; diffusion / flow 60, 2e-4
    patch_size : train on random patches of this size (fine cells; multiple of
        ``factor``) on large domains; sampling always uses the full domain
    zero_threshold : generated values below it are set to 0 (mm/day)
    device : "auto" (cuda, mps, else cpu) or a torch device string
    output_transform : "softplus" (default), "relu" or "linear" for crps/cgan;
        applied both during optimization and sampling. Linear retains the
        historical signed output followed by a final physical projection.
    score_space : "transformed" or "physical" for scoring-rule training.
        Physical uses mm/day before the optional dry-value threshold.
    flow_time_distribution : "uniform" (Lipman objective) or "logit_normal".
    flow_sigma_min : terminal Gaussian noise fraction in the OT path, [0,1).
    edm_sigma_min, edm_sigma_max, edm_rho : Karras sampling schedule.
    edm_p_mean, edm_p_std : mean and SD of log(sigma) for EDM training.
    edm_churn, edm_churn_min, edm_churn_max, edm_noise : Karras stochastic
        churn controls; zero churn gives the deterministic probability-flow ODE.
    content_estimator : "empirical" (Harris content loss), or "fair" (extension
        subtracting sample variance/M; may yield negative finite-ensemble loss).
    missing_indicators : append missing-value indicators to coarse and extra
        predictors, avoiding confusion between missing data and dry weather.
    """

    def __init__(self, method="crps", factor=2, months=(7, 8, 9), time_window=1, width=48, depth=6,
                 noise_channels=4, noise_dim=8, ensemble_size=None, alpha=1.0, energy_weight=0.2, variogram_weight=0.0,
                 pool_scales=(1, 2),
                 content_weight=10.0, n_critic=5, gp_weight=10.0, residual=False, n_steps=24,
                 epochs=None, batch_size=32, lr=None, patience=None, max_minutes=None, patch_size=None,
                 zero_threshold=0.1, coordinates=True, device="auto", seed=0, verbose=False,
                 context="symmetric", output_transform="softplus", score_space="transformed",
                 variogram_fair=True, flow_time_distribution="uniform", flow_sigma_min=0.0,
                 edm_sigma_min=0.002, edm_sigma_max=80.0, edm_rho=7.0,
                 edm_p_mean=-1.2, edm_p_std=1.2, missing_indicators=True,
                 content_estimator="empirical", edm_churn=0.0,
                 edm_churn_min=0.0, edm_churn_max=float("inf"), edm_noise=1.0):
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}")
        if residual and method not in ("diffusion", "flow"):
            raise ValueError("residual=True applies to method='diffusion' or 'flow'")
        if context not in ("symmetric", "past"):
            raise ValueError("context must be 'symmetric' or 'past'")
        if output_transform not in ("softplus", "relu", "linear"):
            raise ValueError("output_transform must be 'softplus', 'relu' or 'linear'")
        if score_space not in ("transformed", "physical"):
            raise ValueError("score_space must be 'transformed' or 'physical'")
        if flow_time_distribution not in ("uniform", "logit_normal"):
            raise ValueError("flow_time_distribution must be 'uniform' or 'logit_normal'")
        if content_estimator not in ("empirical", "fair"):
            raise ValueError("content_estimator must be 'empirical' or 'fair'")
        if not np.isfinite(flow_sigma_min) or not 0 <= flow_sigma_min < 1:
            raise ValueError("flow_sigma_min must be finite and in [0, 1)")
        for name, value in (("edm_sigma_min", edm_sigma_min), ("edm_sigma_max", edm_sigma_max),
                            ("edm_rho", edm_rho), ("edm_p_std", edm_p_std)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if edm_sigma_min >= edm_sigma_max or not np.isfinite(edm_p_mean):
            raise ValueError("EDM requires sigma_min < sigma_max and finite edm_p_mean")
        for name, value in (("edm_churn", edm_churn), ("edm_churn_min", edm_churn_min)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if np.isnan(edm_churn_max) or edm_churn_max < edm_churn_min:
            raise ValueError("edm_churn_max must be >= edm_churn_min")
        if not np.isfinite(edm_noise) or edm_noise <= 0:
            raise ValueError("edm_noise must be finite and positive")
        # Package defaults: (members per sample, epochs, lr, patience).
        d_M, d_ep, d_lr, d_pat = {"crps": (8, 40, 1e-3, 40), "cgan": (8, 30, 2e-4, 30),
                                  "diffusion": (1, 60, 2e-4, 15), "flow": (1, 60, 2e-4, 15)}[method]
        ensemble_size = d_M if ensemble_size is None else ensemble_size
        epochs, lr, patience = (d_ep if epochs is None else epochs), (d_lr if lr is None else lr), (d_pat if patience is None else patience)
        for name, value, minimum in (("factor", factor, 1), ("time_window", time_window, 0),
                                     ("width", width, 1), ("depth", depth, 1),
                                     ("noise_channels", noise_channels, 0), ("noise_dim", noise_dim, 0),
                                     ("ensemble_size", ensemble_size, 1), ("n_critic", n_critic, 1),
                                     ("n_steps", n_steps, 2 if method == "diffusion" else 1),
                                     ("epochs", epochs, 1), ("batch_size", batch_size, 1),
                                     ("patience", patience, 1)):
            _integer(name, value, minimum)
        if ensemble_size < 2 and method in ("crps", "cgan"):
            raise ValueError("the CRPS/content loss needs ensemble_size >= 2")
        if method in ("crps", "cgan") and noise_channels + noise_dim == 0:
            raise ValueError("a stochastic crps/cgan generator needs spatial or global latent noise")
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 (empirical) and 1 (fair CRPS)")
        for name, value in (("energy_weight", energy_weight), ("variogram_weight", variogram_weight),
                            ("content_weight", content_weight), ("gp_weight", gp_weight),
                            ("zero_threshold", zero_threshold)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        if max_minutes is not None and (not np.isfinite(max_minutes) or max_minutes <= 0):
            raise ValueError("max_minutes must be finite and positive")
        if patch_size is not None and (_integer("patch_size", patch_size) % factor or patch_size < factor):
            raise ValueError("patch_size must be a positive multiple of factor")
        pool_scales = tuple(_integer("pool_scales", s) for s in pool_scales)
        if not pool_scales or 1 not in pool_scales:
            raise ValueError("pool_scales must include 1 for the fine-grid CRPS")
        self.method, self.factor, self.months = method, int(factor), validate_months(months)
        self.k, self.width, self.depth = int(time_window), int(width), int(depth)
        self.noise_channels, self.noise_dim = int(noise_channels), int(noise_dim)
        self.M, self.alpha, self.energy_weight = int(ensemble_size), float(alpha), float(energy_weight)
        self.variogram_weight = float(variogram_weight)
        self.pool_scales = tuple(int(s) for s in pool_scales)
        self.content_weight, self.n_critic, self.gp_weight = float(content_weight), int(n_critic), float(gp_weight)
        self.residual, self.n_steps = bool(residual), int(n_steps)
        self.epochs, self.batch_size, self.lr, self.patience = int(epochs), int(batch_size), float(lr), int(patience)
        self.max_minutes, self.patch_size = max_minutes, patch_size
        self.zero_threshold, self.coordinates = float(zero_threshold), bool(coordinates)
        self.device_name, self.seed, self.verbose = device, int(seed), bool(verbose)
        self.context, self.output_transform, self.score_space = context, output_transform, score_space
        self.variogram_fair = bool(variogram_fair)
        self.flow_time_distribution, self.flow_sigma_min = flow_time_distribution, float(flow_sigma_min)
        self.edm_sigma_min, self.edm_sigma_max, self.edm_rho = float(edm_sigma_min), float(edm_sigma_max), float(edm_rho)
        self.edm_p_mean, self.edm_p_std = float(edm_p_mean), float(edm_p_std)
        self.missing_indicators = bool(missing_indicators)
        self.content_estimator = content_estimator
        self.edm_churn, self.edm_noise = float(edm_churn), float(edm_noise)
        self.edm_churn_min, self.edm_churn_max = float(edm_churn_min), float(edm_churn_max)

    # ------------------------------------------------------------------ data
    def _device(self):
        torch = _torch()
        if self.device_name != "auto":
            return torch.device(self.device_name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _t(self, v):
        """mm/day -> network space."""
        return np.log1p(np.maximum(v, 0.0)) / self.scale_

    def _inv(self, z):
        z = np.asarray(z, dtype=np.float64)
        if not np.isfinite(z).all():
            raise FloatingPointError("generative sampler returned nonfinite values; inspect training and noise scales")
        log_amount = np.maximum(z, 0.0) * self.scale_
        if np.max(log_amount, initial=0) >= np.log(np.finfo(np.float32).max):
            raise FloatingPointError("generated precipitation exceeds float32 support; inspect training and EDM scales")
        v = np.expm1(log_amount)
        return np.where(v < self.zero_threshold, 0.0, v)

    def _generator_output(self, z):
        """Physical support in transformed space, shared by training/sampling."""
        torch = _torch()
        if self.output_transform == "softplus":
            return torch.nn.functional.softplus(z)
        if self.output_transform == "relu":
            return z.clamp_min(0)
        return z

    def _score_values(self, z):
        if self.score_space == "transformed":
            return z
        torch = _torch()
        # Do not cap a tail silently: training must surface numerical failures.
        return torch.expm1(z.clamp_min(0) * self.scale_)

    def _field(self, da, name, members=False, check_grid=True):
        """Check labeled arrays before converting them to positional numpy data."""
        allowed = {"T", "Y", "X", "member"} if members else {"T", "Y", "X"}
        if not isinstance(da, xr.DataArray) or not {"T", "Y", "X"}.issubset(da.dims) or set(da.dims) - allowed:
            raise ValueError(f"{name} must be a DataArray with T, Y, X dimensions"
                             + (" and optionally member" if members else ""))
        for dim in ("T", "Y", "X"):
            if dim not in da.coords or da[dim].dims != (dim,) or da.sizes[dim] == 0:
                raise ValueError(f"{name} requires nonempty one-dimensional {dim} coordinates")
        _daily_dates(da["T"].values)
        for dim, expected in (("Y", getattr(self, "y_", None)), ("X", getattr(self, "x_", None))):
            coord = da[dim].values
            if len(np.unique(coord)) != len(coord) or not np.isfinite(coord).all():
                raise ValueError(f"{name} requires unique finite {dim} coordinates")
            if check_grid and expected is not None and not np.array_equal(coord, expected):
                raise ValueError(f"{name} must be on the fitted observation grid; align or regrid first")
        if "member" in da.dims and (da.sizes["member"] == 0 or not da.get_index("member").is_unique):
            raise ValueError(f"{name} must contain nonempty unique member coordinates")
        return da

    def _static(self, static):
        chans, names = [], []
        if self.coordinates:
            lat, lon = np.meshgrid(self.y_, self.x_, indexing="ij")
            for name, a in (("lat", lat), ("lon", lon)):
                sd = a.std()
                chans.append((a - a.mean()) / sd if sd > 0 else np.zeros_like(a)); names.append(name)
        for name, da in (static or {}).items():
            if isinstance(da, xr.DataArray):
                if set(da.dims) != {"Y", "X"} or any(
                        dim not in da.coords or not np.array_equal(da[dim].values, expected)
                        for dim, expected in (("Y", self.y_), ("X", self.x_))):
                    raise ValueError(f"static predictor {name} must be on the fitted observation grid")
            a = np.asarray(da.transpose("Y", "X").values if isinstance(da, xr.DataArray) else da, float)
            if a.shape != (len(self.y_), len(self.x_)) or not np.isfinite(a).any() or np.isinf(a).any():
                raise ValueError(f"static predictor {name} must have the fitted grid shape and finite data")
            sd = np.nanstd(a)
            chans.append(np.nan_to_num((a - np.nanmean(a)) / (sd if sd > 0 else 1.0))); names.append(name)
        H, W = len(self.y_), len(self.x_)
        return (np.stack(chans).astype(np.float32) if chans else np.zeros((0, H, W), np.float32)), names

    def _features(self, fine, dates, extras=None, coarse_mm=None):
        """fine (N, H, W) mm/day of one sequence set -> dict of arrays for batching."""
        if coarse_mm is None:
            coarse_mm = coarse_block_mean(fine, self.factor)
        coarse = self._t(np.nan_to_num(coarse_mm, nan=0.0)).astype(np.float32)
        runs = _runs(dates)
        doy = pd.DatetimeIndex(dates).dayofyear.to_numpy() / 365.25 * 2 * np.pi
        feats = {"coarse": coarse, "coarse_missing": (~np.isfinite(coarse_mm)).astype(np.float32),
                 "lag": _lag_index(runs, self.k, self.context), "runs": runs,
                 "doy": np.stack([np.sin(doy), np.cos(doy)], 1).astype(np.float32)}
        if self.extra_names_:
            if extras is None or set(extras) != set(self.extra_names_):
                raise ValueError(f"predictors {self.extra_names_} are required")
            ex = np.stack([np.asarray(extras[n], float) for n in self.extra_names_], 1)      # (N, E, H, W)
            if ex.shape != (len(fine), len(self.extra_names_), len(self.y_), len(self.x_)):
                raise ValueError("predictors must have the same time and grid shape as precipitation")
            if np.isinf(ex).any():
                raise ValueError("predictors contain infinite values")
            feats["extra"] = np.nan_to_num((ex - self.extra_mean_[None, :, None, None])
                                           / self.extra_sd_[None, :, None, None]).astype(np.float32)
            feats["extra_missing"] = (~np.isfinite(ex)).astype(np.float32)
        return feats

    def _batch(self, feats, idx, crop=None):
        """Network input (B, C, h, w) for days ``idx`` of a feature set."""
        torch = _torch()
        F = torch.nn.functional
        f = self.factor
        c = feats["coarse"][feats["lag"][idx]]                                    # (B, L, hc, wc)
        H, W = len(self.y_), len(self.x_)
        if crop is not None:
            i0, j0, P = crop
            c = c[..., i0 // f:(i0 + P) // f, j0 // f:(j0 + P) // f]
            H = W = P
        x = torch.from_numpy(np.ascontiguousarray(c))
        if f > 1:
            x = F.interpolate(x, scale_factor=f, mode="nearest")
        x = x[..., :H, :W]
        B = len(idx)
        parts = [x, torch.from_numpy(feats["doy"][idx])[:, :, None, None].expand(B, 2, H, W)]
        if self.missing_indicators:
            missing = feats["coarse_missing"][feats["lag"][idx]]
            if crop is not None:
                missing = missing[..., i0 // f:(i0 + P) // f, j0 // f:(j0 + P) // f]
            mt = torch.from_numpy(np.ascontiguousarray(missing))
            if f > 1:
                mt = F.interpolate(mt, scale_factor=f, mode="nearest")
            parts.append(mt[..., :H, :W])
        st = self.static_ if crop is None else self.static_[:, crop[0]:crop[0] + crop[2], crop[1]:crop[1] + crop[2]]
        if st.shape[0]:
            parts.append(torch.from_numpy(st)[None].expand(B, *st.shape))
        if "extra" in feats:
            e = feats["extra"][idx]
            if crop is not None:
                e = e[..., crop[0]:crop[0] + crop[2], crop[1]:crop[1] + crop[2]]
            parts.append(torch.from_numpy(np.ascontiguousarray(e)))
            if self.missing_indicators:
                em = feats["extra_missing"][idx]
                if crop is not None:
                    em = em[..., crop[0]:crop[0] + crop[2], crop[1]:crop[1] + crop[2]]
                parts.append(torch.from_numpy(np.ascontiguousarray(em)))
        return torch.cat(parts, 1).float()

    def _season_data(self, prcp, years, predictors=None):
        dates = pd.DatetimeIndex(np.concatenate([season_dates(int(y), self.months).values for y in years]))
        try:
            fine = prcp.sel(T=dates).transpose("T", "Y", "X").values.astype(float)
            extras = None
            if predictors:
                extras = {n: da.sel(T=dates).transpose("T", "Y", "X").values for n, da in predictors.items()}
        except KeyError as err:
            raise ValueError(f"observations and predictors must cover every day of seasons {list(years)}") from err
        return fine, dates, extras

    # -------------------------------------------------------------- training
    def fit(self, observations, years, predictors=None, static=None, validation_years=None,
            conditioning=None):
        """Fit on observed seasons ``years`` with disjoint validation.

        observations : Dataset with PRCP or DataArray (T, Y, X), mm/day
        predictors : optional {name: DataArray (T, Y, X)} extra conditioning fields on the
            fine grid (e.g. reanalysis TCWV for training, the model's for forecasting)
        static : optional {name: DataArray (Y, X)} (e.g. elevation)
        conditioning : optional DataArray (T, Y, X) of forecast PRCP regridded
            to the observation grid. Supply actual matching forecast valid
            times for paired training; omitted means perfect prognosis on
            observed block means. Unpaired seasonal forecasts are unsuitable.
        validation_years : seasons held out for early stopping (default: the
        last ~15 % of ``years``, at least one). Training and validation seasons
            must be disjoint; at least two seasons are required for the default split.
        """
        prcp = observations["PRCP"] if isinstance(observations, xr.Dataset) else observations
        self._field(prcp, "observations", check_grid=False)
        years = _years(years)
        if validation_years is None:
            if len(years) < 2:
                raise ValueError("at least two seasons are required for disjoint training and validation")
            nv = max(1, int(round(0.15 * len(years))))
            validation_years = years[-nv:]
        validation_years = _years(validation_years, "validation_years")
        train_years = [y for y in years if y not in set(validation_years)]
        if not train_years:
            raise ValueError("training and validation seasons must be disjoint; no training seasons remain")
        self.y_, self.x_ = prcp["Y"].values, prcp["X"].values
        H, W = len(self.y_), len(self.x_)
        if conditioning is not None:
            self._field(conditioning, "conditioning")
        self.training_mode_ = "perfect_prognosis" if conditioning is None else "paired_forecast"
        for name, da in (predictors or {}).items():
            self._field(da, f"predictor {name}")
        self.extra_names_ = sorted(predictors) if predictors else []
        fine, dates, extras = self._season_data(prcp, train_years, predictors)
        if not np.isfinite(fine).any() or np.isinf(fine).any() or np.any(fine[np.isfinite(fine)] < 0):
            raise ValueError("training precipitation must contain finite, nonnegative daily amounts")
        self.valid_mask_ = np.isfinite(fine).any(axis=0)
        self.scale_ = float(np.nanstd(np.log1p(np.maximum(fine, 0)))) or 1.0
        if self.extra_names_:
            if any(not np.isfinite(extras[n]).any() for n in self.extra_names_):
                raise ValueError("each predictor must contain finite training data")
            self.extra_mean_ = np.array([np.nanmean(extras[n]) for n in self.extra_names_])
            self.extra_sd_ = np.array([np.nanstd(extras[n]) or 1.0 for n in self.extra_names_])
        self.static_, self.static_names_ = self._static(static)
        def coarse_condition(selected_years):
            if conditioning is None:
                return None
            cf, _, _ = self._season_data(conditioning, selected_years)
            if np.isinf(cf).any() or np.any(cf[np.isfinite(cf)] < 0):
                raise ValueError("conditioning precipitation must be nonnegative and finite or missing")
            coarse = coarse_block_mean(cf, self.factor)
            if not np.isfinite(coarse).any():
                raise ValueError("conditioning must contain finite training and validation predictors")
            return coarse

        tr = self._features(fine, dates, extras, coarse_mm=coarse_condition(train_years))
        tr["y"] = self._t(np.nan_to_num(fine)).astype(np.float32)
        tr["m"] = np.isfinite(fine).astype(np.float32)
        present = np.repeat(np.repeat(1 - tr["coarse_missing"], self.factor, -2), self.factor, -1)
        tr["m"] *= present[..., :H, :W]
        self.valid_mask_ = tr["m"].any(axis=0)
        if not self.valid_mask_.any():
            raise ValueError("training requires observed targets with finite conditioning block means")
        vf, vd, ve = self._season_data(prcp, validation_years, predictors)
        if not np.isfinite(vf).any() or np.isinf(vf).any() or np.any(vf[np.isfinite(vf)] < 0):
            raise ValueError("validation precipitation must contain finite, nonnegative daily amounts")
        va = self._features(vf, vd, ve, coarse_mm=coarse_condition(validation_years))
        va["y"] = self._t(np.nan_to_num(vf)).astype(np.float32)
        va["m"] = (np.isfinite(vf) & self.valid_mask_[None]).astype(np.float32)
        present = np.repeat(np.repeat(1 - va["coarse_missing"], self.factor, -2), self.factor, -1)
        va["m"] *= present[..., :H, :W]
        if not va["m"].any():
            raise ValueError("validation must contain observations on the training-supported domain")
        offsets = range(-self.k, self.k + 1) if self.context == "symmetric" else range(-self.k, 1)
        self.channels_ = [f"coarse PRCP t{d:+d}" for d in offsets] + ["doy sin", "doy cos"]
        if self.missing_indicators:
            self.channels_ += [f"coarse missing t{d:+d}" for d in offsets]
        self.channels_ += self.static_names_ + self.extra_names_
        if self.missing_indicators:
            self.channels_ += [f"{n} missing" for n in self.extra_names_]
        # A corrector fitted on an earlier data set is invalid after refitting.
        self.predictor_corrector_ = None
        torch = _torch()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        in_ch = len(self.channels_)
        P = self.patch_size
        if P is not None:
            P = int(P) // self.factor * self.factor
            if P >= min(H, W):
                P = None
        self.patch_ = P
        maxd = max(1, min(H, W) // 2) if P is None else max(1, P // 2)
        nets = _nets()
        dev = self._device()
        self.device_ = dev
        if self.method in ("crps", "cgan"):
            self.net_ = nets["Generator"](in_ch, self.noise_channels, self.noise_dim, self.width, self.depth, maxd).to(dev)
            if self.method == "cgan":
                self.critic_ = nets["Critic"](in_ch, self.width, max(3, self.depth // 2), maxd).to(dev)
        else:
            if self.residual:
                self.mean_net_ = nets["Backbone"](in_ch, 1, self.width, self.depth, 0, maxd).to(dev)
            self.net_ = nets["Denoiser"](in_ch, self.width, self.depth, maxd).to(dev)
        t0 = time.time()
        self.history_ = []
        if self.method in ("diffusion", "flow") and self.residual:
            self._train(tr, va, rng, t0, stage="mean")
            self.sigma_data_ = max(self._residual_sd(tr), 1e-3)
        elif self.method in ("diffusion", "flow"):
            self.sigma_data_ = max(float(tr["y"][tr["m"] > 0].std()), 1e-3)
        self._train(tr, va, rng, t0, stage="main")
        self.history_ = pd.DataFrame(self.history_)
        self.train_seconds_ = time.time() - t0
        self.train_years_, self.validation_years_ = train_years, list(validation_years)
        return self

    def _crop(self, rng):
        if self.patch_ is None:
            return None
        P, f = self.patch_, self.factor
        H, W = len(self.y_), len(self.x_)
        i0 = rng.integers(0, (H - P) // f + 1) * f
        j0 = rng.integers(0, (W - P) // f + 1) * f
        return int(i0), int(j0), P

    def _targets(self, data, idx, crop):
        torch = _torch()
        y, m = data["y"][idx], data["m"][idx]
        if crop is not None:
            i0, j0, P = crop
            y, m = y[:, i0:i0 + P, j0:j0 + P], m[:, i0:i0 + P, j0:j0 + P]
        return torch.from_numpy(np.ascontiguousarray(y))[:, None], torch.from_numpy(np.ascontiguousarray(m))[:, None]

    def _residual_sd(self, data):
        torch = _torch()
        res, w = [], []
        with torch.no_grad():
            for idx in np.array_split(np.arange(len(data["y"])), max(1, len(data["y"]) // 256)):
                x = self._batch(data, idx).to(self.device_)
                mu = self.mean_net_(x).cpu().numpy()[:, 0]
                res.append(((data["y"][idx] - mu) * data["m"][idx]).ravel()); w.append(data["m"][idx].ravel())
        r, w = np.concatenate(res), np.concatenate(w)
        return float(np.sqrt((r ** 2 * w).sum() / w.sum()))

    def _train(self, tr, va, rng, t0, stage):
        torch = _torch()
        dev = self.device_
        if stage == "mean":
            params, step = self.mean_net_.parameters(), self._step_mean
        else:
            params, step = self.net_.parameters(), {"crps": self._step_crps, "cgan": self._step_gan,
                                                     "diffusion": self._step_edm, "flow": self._step_flow}[self.method]
        betas = (0.5, 0.9) if self.method == "cgan" and stage == "main" else (0.9, 0.999)
        self.opt_ = torch.optim.Adam(params, lr=self.lr, betas=betas)
        if self.method == "cgan" and stage == "main":
            self.opt_c_ = torch.optim.Adam(self.critic_.parameters(), lr=self.lr, betas=betas)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_, T_max=max(1, self.epochs), eta_min=self.lr / 20)
        net = self.mean_net_ if stage == "mean" else self.net_
        best, best_state, bad = np.inf, None, 0
        N = len(tr["y"])
        epochs = self.epochs
        for ep in range(epochs):
            order = rng.permutation(N)
            losses = []
            net.train()
            for b in range(0, N, self.batch_size):
                idx = order[b:b + self.batch_size]
                crop = self._crop(rng)
                x = self._batch(tr, idx, crop).to(dev)
                y, m = (t.to(dev) for t in self._targets(tr, idx, crop))
                supported = m.flatten(1).sum(1) > 0
                if not supported.any():
                    continue
                x, y, m = x[supported], y[supported], m[supported]
                value = step(x, y, m)
                if not np.isfinite(value):
                    raise FloatingPointError(f"nonfinite {self.method}/{stage} training loss at epoch {ep}")
                losses.append(value)
            if not losses:
                raise ValueError("no observed targets in sampled training patches; disable patch_size or revise mask")
            sched.step()
            net.eval()
            v = self._validate(va, stage)
            if not np.isfinite(v):
                raise FloatingPointError(f"nonfinite {self.method}/{stage} validation loss at epoch {ep}")
            self.history_.append({"stage": stage, "epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": v,
                                  "minutes": (time.time() - t0) / 60})
            if self.verbose:
                print(f"{self.method}/{stage} epoch {ep:3d} train {np.mean(losses):.4f} val {v:.4f}")
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                bad += 1
            if bad >= self.patience or (self.max_minutes and (time.time() - t0) / 60 > self.max_minutes):
                break
        if best_state is not None:
            net.load_state_dict(best_state)
        if stage == "main":
            self.best_val_loss_ = best

    # --- one optimisation step per method
    def _noise(self, B, H, W, gen=None):
        torch = _torch()
        zs = torch.randn(B, self.noise_channels, H, W, generator=gen, device="cpu").to(self.device_)
        zg = torch.randn(B, self.noise_dim, generator=gen, device="cpu").to(self.device_)
        return zs, zg

    def _ensemble(self, x, M, gen=None):
        B, _, H, W = x.shape
        xr_ = x.repeat_interleave(M, 0)
        zs, zg = self._noise(B * M, H, W, gen)
        return self._generator_output(self.net_(xr_, zs, zg)).view(B, M, H, W)

    def _crps_loss(self, ens, y, m):
        torch = _torch()
        y, m = y[:, 0], m[:, 0]
        ens, y = self._score_values(ens), self._score_values(y)
        loss = 0.0
        for s in self.pool_scales:
            if s > 1 and min(y.shape[-2:]) < s:
                continue
            e_s = _masked_pool(torch, ens.flatten(0, 1)[:, None], m.repeat_interleave(ens.shape[1], 0)[:, None], s)[0]
            e_s = e_s.view(ens.shape[0], ens.shape[1], *e_s.shape[-2:])
            y_s, m_s = _masked_pool(torch, y[:, None], m[:, None], s)
            loss = loss + _afcrps(torch, e_s, y_s[:, 0], m_s[:, 0], self.alpha)
        if self.energy_weight:
            loss = loss + self.energy_weight * _energy_score(torch, ens, y, m)
        if self.variogram_weight:
            loss = loss + self.variogram_weight * _variogram_score(torch, ens, y, m, fair=self.variogram_fair)
        return loss

    def _step_crps(self, x, y, m):
        loss = self._crps_loss(self._ensemble(x, self.M), y, m)
        self.opt_.zero_grad(); loss.backward(); self.opt_.step()
        return float(loss.detach())

    def _step_gan(self, x, y, m):
        """WGAN-GP step after Gulrajani, Ahmed, Arjovsky, Dumoulin and Courville
        (2017), https://arxiv.org/abs/1704.00028, with Harris and coauthors'
        (2022) content mechanism. Missing-cell masking is a package extension.
        """
        torch = _torch()
        B, _, H, W = x.shape
        for _ in range(self.n_critic):
            with torch.no_grad():
                zs, zg = self._noise(B, H, W)
                fake = self._generator_output(self.net_(x, zs, zg)) * m
            real = y * m
            eps = torch.rand(B, 1, 1, 1, device=x.device)
            mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
            g = torch.autograd.grad(self.critic_(x, mix).sum(), mix, create_graph=True)[0]
            # Norm on the observed subspace: missing cells are not part of
            # the transport distance and must not relax the penalty.
            gp = (((g * m).flatten(1).norm(dim=1) - 1) ** 2).mean()
            lc = self.critic_(x, fake).mean() - self.critic_(x, real).mean() + self.gp_weight * gp
            self.opt_c_.zero_grad(); lc.backward(); self.opt_c_.step()
        ens = self._ensemble(x, self.M)
        adv = -self.critic_(x, ens[:, :1] * m).mean()
        content = self._content_loss(ens, y, m)
        loss = adv + self.content_weight * content
        self.opt_.zero_grad(); loss.backward(); self.opt_.step()
        return float(content.detach())

    def _content_loss(self, ens, y, m):
        """Ensemble-mean content loss after Lucy Harris, Andrew T. T. McRae,
        Matthew Chantry, Peter D. Dueben and Tim N. Palmer (2022).

        Reference: https://doi.org/10.1029/2022MS003120. The optional ``fair``
        estimator subtracts sample variance/M under independent draws; this
        package extension targets population-mean MSE and can be negative.
        """
        content_term = (ens.mean(1, keepdim=True) - y) ** 2
        if self.content_estimator == "fair":
            content_term = content_term - ens.var(1, unbiased=True, keepdim=True) / ens.shape[1]
        return (content_term * m).sum() / m.sum().clamp_min(1.0)

    def _step_mean(self, x, y, m):
        """Regression stage of Mardani and coauthors' (2023/2025) residual
        construction, https://doi.org/10.1038/s43247-025-02042-5, using the
        package network rather than the original CorrDiff architecture.
        """
        loss = (((self.mean_net_(x) - y) ** 2) * m).sum() / m.sum().clamp_min(1.0)
        self.opt_.zero_grad(); loss.backward(); self.opt_.step()
        return float(loss.detach())

    def _target(self, x, y):
        if self.residual:
            torch = _torch()
            with torch.no_grad():
                return y - self.mean_net_(x)
        return y

    def _edm_D(self, yn, x, sigma):
        """EDM skip/input/output/noise preconditioning of Tero Karras, Miika
        Aittala, Timo Aila and Samuli Laine (2022),
        https://arxiv.org/abs/2206.00364, applied to a package conditional net.
        """
        sd = self.sigma_data_
        s = sigma.view(-1, 1, 1, 1)
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * sd / (s ** 2 + sd ** 2).sqrt()
        c_in = 1 / (s ** 2 + sd ** 2).sqrt()
        return c_skip * yn + c_out * self.net_(c_in * yn, x, sigma.log() / 4)

    def _step_edm(self, x, y, m):
        """Log-normal noise sampling and weighted EDM denoising objective of
        Karras, Aittala, Aila and Laine (2022),
        https://arxiv.org/abs/2206.00364, with observed-cell masking.
        """
        torch = _torch()
        t = self._target(x, y)
        sigma = (torch.randn(len(x), device=x.device) * self.edm_p_std + self.edm_p_mean).exp()
        n = torch.randn_like(t) * sigma.view(-1, 1, 1, 1)
        w = (sigma ** 2 + self.sigma_data_ ** 2) / (sigma * self.sigma_data_) ** 2
        loss = (w.view(-1, 1, 1, 1) * (self._edm_D(t + n, x, sigma) - t) ** 2 * m).sum() / m.sum().clamp_min(1.0)
        self.opt_.zero_grad(); loss.backward(); self.opt_.step()
        return float(loss.detach())

    def _step_flow(self, x, y, m):
        """Gaussian straight-path conditional flow matching after Yaron Lipman,
        Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel and Matt Le (2023),
        https://arxiv.org/abs/2210.02747. Uniform t is the paper objective;
        logit-normal t and residual targets are explicit package variants.
        """
        torch = _torch()
        t1 = self._target(x, y)
        z = torch.randn_like(t1) * self.sigma_data_
        tt = (torch.rand(len(x), device=x.device) if self.flow_time_distribution == "uniform"
              else torch.sigmoid(torch.randn(len(x), device=x.device)))
        b = 1 - self.flow_sigma_min
        xt = (1 - b * tt.view(-1, 1, 1, 1)) * z + tt.view(-1, 1, 1, 1) * t1
        v = self.net_(xt, x, tt)
        loss = (((v - (t1 - b * z)) ** 2) * m).sum() / m.sum().clamp_min(1.0) / self.sigma_data_ ** 2
        self.opt_.zero_grad(); loss.backward(); self.opt_.step()
        return float(loss.detach())

    def _validate(self, va, stage):
        """Validation loss: CRPS of M members (crps), 4 members (cgan), MSE (mean
        net) or the denoising / flow loss on fixed noise draws (diffusion, flow)."""
        torch = _torch()
        gen = torch.Generator().manual_seed(1234)
        idx_all = np.arange(len(va["y"]))
        tot, n = 0.0, 0
        with torch.no_grad():
            for idx in np.array_split(idx_all, max(1, len(idx_all) // 128)):
                x = self._batch(va, idx).to(self.device_)
                y, m = (t.to(self.device_) for t in self._targets(va, idx, None))
                supported = m.flatten(1).sum(1) > 0
                if not supported.any():
                    continue
                x, y, m = x[supported], y[supported], m[supported]
                if stage == "mean":
                    v = (((self.mean_net_(x) - y) ** 2) * m).sum() / m.sum().clamp_min(1.0)
                elif self.method == "crps":
                    v = self._crps_loss(self._ensemble(x, self.M, gen), y, m)
                elif self.method == "cgan":
                    ens = self._ensemble(x, 4, gen)
                    v = _afcrps(torch, self._score_values(ens), self._score_values(y[:, 0]), m[:, 0], 1.0)
                elif self.method == "diffusion":
                    t = self._target(x, y)
                    sigma = (torch.randn(len(x), generator=gen) * self.edm_p_std + self.edm_p_mean).exp().to(x.device)
                    nz = torch.randn(t.shape, generator=gen).to(x.device) * sigma.view(-1, 1, 1, 1)
                    w = (sigma ** 2 + self.sigma_data_ ** 2) / (sigma * self.sigma_data_) ** 2
                    v = (w.view(-1, 1, 1, 1) * (self._edm_D(t + nz, x, sigma) - t) ** 2 * m).sum() / m.sum().clamp_min(1.0)
                else:
                    t1 = self._target(x, y)
                    z = torch.randn(t1.shape, generator=gen).to(x.device) * self.sigma_data_
                    tt = (torch.rand(len(x), generator=gen) if self.flow_time_distribution == "uniform"
                          else torch.sigmoid(torch.randn(len(x), generator=gen))).to(x.device)
                    b = 1 - self.flow_sigma_min
                    ts = tt.view(-1, 1, 1, 1)
                    xt = (1 - b * ts) * z + ts * t1
                    v = (((self.net_(xt, x, tt) - (t1 - b * z)) ** 2) * m).sum() / m.sum().clamp_min(1.0) / self.sigma_data_ ** 2
                tot += float(v) * len(x); n += len(x)
        if not n:
            raise ValueError("validation has no observed target samples")
        return tot / max(n, 1)

    # -------------------------------------------------------------- sampling
    def _sample_batch(self, x, zs=None, zg=None, z0=None, churn_seeds=None):
        """One sample per row of x, from given noise (network space)."""
        torch = _torch()
        with torch.no_grad():
            if self.method in ("crps", "cgan"):
                return self._generator_output(self.net_(x, zs, zg))[:, 0]
            mu = self.mean_net_(x) if self.residual else 0.0
            if self.method == "diffusion":
                smin, smax, rho, N = self.edm_sigma_min, self.edm_sigma_max, self.edm_rho, self.n_steps
                i = torch.arange(N, device=x.device, dtype=torch.float32)
                sig = (smax ** (1 / rho) + i / (N - 1) * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
                sig = torch.cat([sig, torch.zeros(1, device=x.device)])
                y = z0 * sig[0]
                for j in range(N):
                    s, s2 = sig[j], sig[j + 1]
                    gamma = min(self.edm_churn / N, math.sqrt(2) - 1) if self.edm_churn_min <= s <= self.edm_churn_max else 0.0
                    sh = s * (1 + gamma)
                    yh = y
                    if gamma:
                        if churn_seeds is None:
                            raise ValueError("stochastic EDM sampling requires reproducible churn_seeds")
                        noises = np.stack([np.random.default_rng(int(seed)).standard_normal(tuple(y.shape[1:]))
                                           for seed in churn_seeds[j]]).astype(np.float32)
                        nz = torch.from_numpy(noises).to(x.device)
                        yh = y + (sh ** 2 - s ** 2).sqrt() * self.edm_noise * nz
                    d = (yh - self._edm_D(yh, x, sh.expand(len(x)))) / sh
                    y2 = yh + (s2 - sh) * d
                    if s2 > 0:
                        d2 = (y2 - self._edm_D(y2, x, s2.expand(len(x)))) / s2
                        y2 = yh + (s2 - sh) * (d + d2) / 2
                    y = y2
            else:
                y = z0 * self.sigma_data_
                ts = torch.linspace(0, 1, self.n_steps + 1, device=x.device)
                for j in range(self.n_steps):
                    t, t2 = ts[j], ts[j + 1]
                    v = self.net_(y, x, t.expand(len(x)))
                    y2 = y + (t2 - t) * v
                    # Flow time t=1 is nonsingular. Unlike EDM sigma=0,
                    # the final interval also needs the Heun correction.
                    v2 = self.net_(y2, x, t2.expand(len(x)))
                    y2 = y + (t2 - t) * (v + v2) / 2
                    y = y2
            return (y + mu)[:, 0]

    def _sample_sequences(self, feats, n_seq, T, n_samples, rng, noise_rho, batch_days):
        """feats built on n_seq sequences of T days (row = seq * T + t) -> (n_seq*n_samples, T, H, W)."""
        torch = _torch()
        H, W = len(self.y_), len(self.x_)
        out = np.empty((n_seq * n_samples, T, H, W), np.float32)
        runs = feats["runs"][:T]
        for q in range(n_seq):
            for s in range(n_samples):
                if self.method in ("crps", "cgan"):
                    zs_all = ar1_noise(rng, (T, self.noise_channels, H, W), noise_rho, runs)
                    zg_all = ar1_noise(rng, (T, self.noise_dim), noise_rho, runs)
                else:
                    z0_all = ar1_noise(rng, (T, 1, H, W), noise_rho, runs)
                    churn_seeds = (rng.integers(0, np.iinfo(np.int64).max, (self.n_steps, T))
                                   if self.method == "diffusion" and self.edm_churn else None)
                for b in range(0, T, batch_days):
                    sl = slice(b, min(T, b + batch_days))
                    idx = q * T + np.arange(sl.start, sl.stop)
                    x = self._batch(feats, idx).to(self.device_)
                    if self.method in ("crps", "cgan"):
                        z = self._sample_batch(x, torch.from_numpy(zs_all[sl]).to(self.device_),
                                               torch.from_numpy(zg_all[sl]).to(self.device_))
                    else:
                        z = self._sample_batch(x, z0=torch.from_numpy(z0_all[sl]).to(self.device_),
                                               churn_seeds=churn_seeds[:, sl] if churn_seeds is not None else None)
                    values = z.cpu().numpy()
                    if hasattr(self, "valid_mask_"):
                        values = np.where(self.valid_mask_[None], values, 0.0)
                    out[q * n_samples + s, sl] = self._inv(values)
        return out

    def fit_predictor_correction(self, model, observations, years, method="loci_qm", wet_threshold=0.1,
                                 allow_validation_years=False):
        """Adjust the coarse predictor of the model to the observed one (perfect-prognosis
        consistency).

        A bias-corrected model can have the right daily distribution at every fine
        cell and still the wrong distribution of *block means*, if its fields are
        too smooth or too patchy. The generator was trained on observed block
        means, so it would then see unfamiliar inputs. This fits, per coarse cell and
        calendar month, a ``DailyBiasCorrector`` from the block means of ``model``
        (member, T, Y, X; the corrected hindcast of ``years``) to the block means
        of the observations. ``downscale`` then applies it (``correct_predictors``)."""
        from .dynamical import DailyBiasCorrector, season_blocks
        if not hasattr(self, "net_"):
            raise RuntimeError("Call fit before fit_predictor_correction")
        prcp = observations["PRCP"] if isinstance(observations, xr.Dataset) else observations
        self._field(model, "model", members=True)
        self._field(prcp, "observations")
        years = _years(years)
        overlap = set(years) & set(getattr(self, "validation_years_", []))
        if overlap and not allow_validation_years:
            raise ValueError("predictor correction would use held-out validation years; calibrate on train_years_ "
                             "or explicitly set allow_validation_years=True for non-independent diagnostics")
        H, W = len(self.y_), len(self.x_)
        mod = season_blocks(model, years, self.months)                           # (Y, M, T, S)
        ob = season_blocks(prcp, years, self.months)[:, 0]                       # (Y, T, S)
        cm = coarse_block_mean(mod.reshape(*mod.shape[:3], H, W), self.factor)
        co = coarse_block_mean(ob.reshape(*ob.shape[:2], H, W), self.factor)
        month = season_dates(years[0], self.months).month.to_numpy()
        self.coarse_shape_ = cm.shape[-2:]
        self.predictor_corrector_ = DailyBiasCorrector(method, wet_threshold).fit(
            cm.reshape(*cm.shape[:3], -1), co.reshape(*co.shape[:2], -1), month)
        return self

    def downscale(self, model, predictors=None, n_samples=1, noise_rho=0.0, seed=None, batch_days=256,
                  correct_predictors=True, conserve_coarse_mean=False):
        """Generate fine fields from (bias-corrected) model daily precipitation.

        model : DataArray (member, T, Y, X) or (T, Y, X) on the fine grid (regrid
            first, e.g. nearest); it is block-averaged by ``factor`` like the
            training predictors. Any dates; lags stay inside contiguous runs.
        predictors : {name: DataArray (member, T, Y, X) or (T, Y, X)} if fitted with extras
        n_samples : generated fields per model member (output member = m * n_samples + s)
        noise_rho : lag-1 correlation of the latent noise along days (0 = independent)
        correct_predictors : apply ``fit_predictor_correction`` when it was fitted
        conserve_coarse_mean : optional nonnegative block normalization to
            exactly preserve the conditioning block mean over valid output
            cells. This is a package extension, unsuitable when forecast
            block totals are biased. If a generated block is entirely dry
            but the required mean is positive, spread its rain uniformly.
            Conservation follows zero_threshold and can reintroduce values
            smaller than that threshold.
        """
        if not hasattr(self, "net_"):
            raise RuntimeError("Call fit before downscale")
        n_samples = _integer("n_samples", n_samples)
        batch_days = _integer("batch_days", batch_days)
        if not np.isfinite(noise_rho) or not -1 < noise_rho < 1:
            raise ValueError("noise_rho must be finite and strictly between -1 and 1")
        self._field(model, "model", members=True)
        if "member" not in model.dims:
            model = model.expand_dims(member=[0])
        model = model.transpose("member", "T", "Y", "X")
        if not (np.array_equal(model["Y"].values, self.y_) and np.array_equal(model["X"].values, self.x_)):
            raise ValueError("model fields must be on the fitted (observation) grid; regrid first")
        Mm, T = model.sizes["member"], model.sizes["T"]
        dates = pd.DatetimeIndex(model["T"].values)
        fine = model.values.astype(float).reshape(Mm * T, len(self.y_), len(self.x_))
        if np.isinf(fine).any() or np.any(fine[np.isfinite(fine)] < 0):
            raise ValueError("model precipitation must contain finite or missing, nonnegative daily amounts")
        all_dates = pd.DatetimeIndex(np.tile(dates.values, Mm))
        extras = None
        if self.extra_names_:
            if not predictors or set(predictors) != set(self.extra_names_):
                raise ValueError(f"predictors {self.extra_names_} are required")
            extras = {}
            for n in self.extra_names_:
                da = predictors[n]
                self._field(da, f"predictor {n}", members=True)
                if not np.array_equal(da["T"].values, model["T"].values):
                    raise ValueError(f"predictor {n} must use the same T coordinates as model")
                if "member" in da.dims and not np.array_equal(da["member"].values, model["member"].values):
                    raise ValueError(f"predictor {n} must use the same member coordinates as model")
                if "member" not in da.dims:
                    da = da.expand_dims(member=model["member"].values)
                extras[n] = da.transpose("member", "T", "Y", "X").values.reshape(Mm * T, len(self.y_), len(self.x_))
        coarse_mm = coarse_block_mean(fine, self.factor)
        if correct_predictors and getattr(self, "predictor_corrector_", None) is not None:
            hc, wc = coarse_mm.shape[-2:]
            arr = coarse_mm.reshape(Mm, T, hc * wc)
            fixed = self.predictor_corrector_.transform(arr, dates.month.to_numpy())
            coarse_mm = np.where(np.isfinite(fixed), fixed, arr).reshape(Mm * T, hc, wc)
        feats = self._features(fine, all_dates, extras, coarse_mm=coarse_mm)
        # lags must not cross members: rebuild lag index per member
        lag = _lag_index(_runs(dates), self.k, self.context)
        feats["lag"] = np.concatenate([lag + q * T for q in range(Mm)])
        feats["runs"] = np.tile(_runs(dates), Mm)
        rng = np.random.default_rng(self.seed + 7 if seed is None else seed)
        out = self._sample_sequences(feats, Mm, T, n_samples, rng, float(noise_rho), batch_days)
        if hasattr(self, "valid_mask_"):
            out = np.where(self.valid_mask_[None, None], out, np.nan)
        f = self.factor
        target = coarse_mm.reshape(Mm, T, *coarse_mm.shape[-2:])
        target = np.repeat(target, n_samples, axis=0)
        predictor_valid = np.repeat(np.repeat(np.isfinite(target), f, -2), f, -1)
        out = np.where(predictor_valid[..., :len(self.y_), :len(self.x_)], out, np.nan)
        fallback = 0
        if conserve_coarse_mean:
            out, fallback = self._conserve_blocks(out, target)
        self.mass_conservation_fallback_blocks_ = fallback
        da = xr.DataArray(out, dims=("member", "T", "Y", "X"),
                          coords={"member": np.arange(out.shape[0]), "T": dates, "Y": self.y_, "X": self.x_,
                                  "model_member": ("member", np.repeat(model["member"].values, n_samples)),
                                  "stochastic_sample": ("member", np.tile(np.arange(n_samples), Mm))},
                          name="PRCP", attrs={"units": "mm d-1"})
        ds = xr.Dataset({"PRCP": da})
        ds.attrs.update(generator=f"was-disaggregation GenerativeDownscaler ({self.method}{', residual' if self.residual else ''})",
                        factor=self.factor, noise_rho=float(noise_rho),
                        training_mode=getattr(self, "training_mode_", "perfect_prognosis"),
                        context=self.context, score_space=self.score_space,
                        output_transform=self.output_transform,
                        coarse_mean_conserved=int(bool(conserve_coarse_mean)),
                        conservation_fallback_blocks=fallback)
        return ds

    def _conserve_blocks(self, output, target):
        """Match equal-cell block means; support NaNs and partial edge blocks."""
        output = np.asarray(output, dtype=np.float64).copy()
        if np.isinf(target).any() or np.any(target[np.isfinite(target)] < 0):
            raise ValueError("conditioning means must be nonnegative and finite or missing")
        fallback, f = 0, self.factor
        for i in range(target.shape[-2]):
            for j in range(target.shape[-1]):
                block = output[..., i * f:(i + 1) * f, j * f:(j + 1) * f]
                valid = np.isfinite(block)
                n = valid.sum((-2, -1))
                total = np.where(valid, block, 0).sum((-2, -1))
                requested = target[..., i, j]
                mass = requested * n
                multiplier = np.divide(mass, total, out=np.zeros_like(total), where=total > 0)
                fixed = block * multiplier[..., None, None]
                dry = (total == 0) & (n > 0) & np.isfinite(requested) & (requested > 0)
                fallback += int(dry.sum())
                fixed = np.where(dry[..., None, None] & valid, requested[..., None, None], fixed)
                output[..., i * f:(i + 1) * f, j * f:(j + 1) * f] = np.where(valid, fixed, np.nan)
        if np.any(output[np.isfinite(output)] > np.finfo(np.float32).max):
            raise FloatingPointError("coarse conservation exceeds float32 support")
        return output.astype(np.float32), fallback

    def sample_perfect_prognosis(self, observations, years, n_samples=10, seed=None, noise_rho=0.0,
                                predictors=None):
        """Downscale the block-averaged *observations* of ``years`` (daily verification
        of the perfect-prognosis mapping on held-out seasons). Supply observed
        ``predictors`` when extra fields were used in training. Model predictor
        bias correction does not apply to these observed conditioning fields.
        """
        prcp = observations["PRCP"] if isinstance(observations, xr.Dataset) else observations
        dates = pd.DatetimeIndex(np.concatenate([season_dates(y, self.months).values for y in _years(years)]))
        extras = {name: da.sel(T=dates) for name, da in predictors.items()} if predictors else None
        return self.downscale(prcp.sel(T=dates), predictors=extras, n_samples=n_samples, seed=seed,
                              noise_rho=noise_rho, correct_predictors=False)

    # ----------------------------------------------------------------- I/O
    def save(self, path):
        """Save a fitted model. Only load checkpoint files from trusted sources."""
        if not hasattr(self, "net_"):
            raise RuntimeError("Call fit before save")
        torch = _torch()
        state = {"init": {k: v for k, v in self.__dict__.items() if not k.endswith("_")},
                 "fitted": {k: v for k, v in self.__dict__.items()
                            if k.endswith("_") and k not in ("net_", "critic_", "mean_net_", "opt_", "opt_c_", "device_", "history_")},
                 "net": self.net_.state_dict(),
                 "mean_net": self.mean_net_.state_dict() if getattr(self, "mean_net_", None) is not None else None}
        torch.save(state, path)

    @classmethod
    def load(cls, path, device="auto"):
        """Load a trusted checkpoint (the format uses Python pickle)."""
        torch = _torch()
        st = torch.load(path, weights_only=False, map_location="cpu")
        obj = cls.__new__(cls)
        obj.__dict__.update(st["init"]); obj.__dict__.update(st["fitted"])
        # Preserve the behavior and channel count of trusted pre-0.10 files.
        legacy = {"context": "symmetric", "output_transform": "linear", "score_space": "transformed",
                  "variogram_fair": False, "flow_time_distribution": "logit_normal", "flow_sigma_min": 0.0,
                  "edm_sigma_min": 0.002, "edm_sigma_max": 80.0, "edm_rho": 7.0,
                  "edm_p_mean": -1.2, "edm_p_std": 1.2, "missing_indicators": False,
                  "content_estimator": "empirical", "edm_churn": 0.0, "edm_noise": 1.0,
                  "edm_churn_min": 0.0, "edm_churn_max": float("inf")}
        for name, default in legacy.items():
            obj.__dict__.setdefault(name, default)
        obj.device_name = device
        dev = obj._device()
        obj.device_ = dev
        nets = _nets()
        in_ch = len(obj.channels_)
        H, W = len(obj.y_), len(obj.x_)
        maxd = max(1, min(H, W) // 2) if obj.patch_ is None else max(1, obj.patch_ // 2)
        if obj.method in ("crps", "cgan"):
            obj.net_ = nets["Generator"](in_ch, obj.noise_channels, obj.noise_dim, obj.width, obj.depth, maxd)
        else:
            obj.net_ = nets["Denoiser"](in_ch, obj.width, obj.depth, maxd)
            if obj.residual:
                obj.mean_net_ = nets["Backbone"](in_ch, 1, obj.width, obj.depth, 0, maxd)
                obj.mean_net_.load_state_dict(st["mean_net"]); obj.mean_net_.to(dev).eval()
        obj.net_.load_state_dict(st["net"]); obj.net_.to(dev).eval()
        return obj
