"""Scalable, reproducible latent Gaussian spatial dependence.

Kernels use Euclidean chord distance on a sphere, *not* great-circle distance.
This small regional-distance approximation guarantees positive-definite kernels
without constructing a site-by-site covariance matrix. The power kernel is
``(1 + distance / range_km)**(-alpha)`` (the sign of the exponent in the source
notation is negative); the exponential kernel is ``exp(-distance/range_km)``.

Spatial correlations of observed rain occurrence/amounts are not interchangeable
with correlations of the latent Gaussian draws. Binary observations can be
calibrated by bivariate-normal threshold inversion; continuous observations use
Gaussian rank scores. Conditional Markov occurrence thresholds, Gamma marginal
transforms, local VAR filters, and finite Fourier features introduce additional
approximations. Generated physical correlations therefore still need validation.

Scientific attribution
----------------------
* Wilks, D. S. (1998), "Multisite generalization of a daily stochastic
  precipitation generation model", Journal of Hydrology 210(1-4), 178-191,
  https://doi.org/10.1016/S0022-1694(98)00186-3: correlated latent Gaussian
  random streams for multisite rainfall occurrence and amounts.
* Wilks, D. S. (2002), "Realizations of daily weather in forecast seasonal
  climate", Journal of Hydrometeorology 3(2), 195-207,
  https://doi.org/10.1175/1525-7541(2002)003<0195:RODWIF>2.0.CO;2:
  distance-based occurrence/amount dependence and climatological spatial
  dependence held fixed under seasonal conditioning.
* Plackett, R. L. (1954), "A reduction formula for normal multivariate
  integrals", Biometrika 41(3-4), 351-360,
  https://doi.org/10.1093/biomet/41.3-4.351: the bivariate-normal reduction
  identity numerically integrated in ``_bivariate_normal_cdf``.
* Rahimi, A., and Recht, B. (2007), "Random Features for Large-Scale Kernel
  Machines", Advances in Neural Information Processing Systems 20,
  https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html:
  paired sine/cosine Fourier features approximate a shift-invariant kernel.
  The sphere embedding, spectral distributions and tile-stable draw scheme
  used here are a weather-generator implementation choice.

``GaussianSpatialField`` is a scalable kernel approximation with exact standard
Gaussian marginal variance. ``DenseGaussianSpatialField`` uses the specified
finite-domain covariance to numerical factorization precision. Dense Gaussian
sampling is a standard linear-algebra construction, not a separate published
physical weather model. Neither backend claims exact physical rainfall
correlations, conditional Markov cross-site associations, tail dependence or
reproduction of the Wilks station experiments. Chord-distance kernels and
pair-sampling calibration are explicit package extensions.
"""

from __future__ import annotations

import hashlib
import warnings

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata

EARTH_RADIUS_KM = 6371.0088
_UINT64_MASK = (1 << 64) - 1
_QUAD_X, _QUAD_W = leggauss(64)


def _coordinates(lat, lon):
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if lat.ndim != 1 or lon.shape != lat.shape:
        raise ValueError("lat and lon must be equally sized one-dimensional arrays")
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("Coordinates must be finite")
    if np.any(np.abs(lat) > 90):
        raise ValueError("Latitude must be between -90 and 90 degrees")
    lon = (lon + 180.0) % 360.0 - 180.0
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    xyz = EARTH_RADIUS_KM * np.column_stack(
        (np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi))
    )
    return lat, lon, xyz


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _seed(seed, namespace, step=0, stream=0):
    # SeedSequence accepts arbitrary nonnegative Python integers. Separate fixed
    # namespaces keep feature, coefficient, pair-selection and independent paths
    # unrelated even when the user chooses the same master seed.
    signed_step = 2 * step if step >= 0 else -2 * step - 1
    return np.random.SeedSequence([seed, namespace, signed_step, stream])


def kernel_correlation(distance_km, model):
    """Return the fitted *latent* correlation at a chord distance in kilometres."""
    distance = np.asarray(distance_km, dtype=float)
    if np.any(distance < 0):
        raise ValueError("Distances must be nonnegative")
    length = float(model.get("range_km", 150.0))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("range_km must be positive and finite")
    kind = model.get("kind", "exponential")
    if kind == "exponential":
        return np.exp(-distance / length)
    if kind == "power":
        alpha = float(model.get("alpha", 1.0))
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        return (1.0 + distance / length) ** (-alpha)
    raise ValueError("kind must be 'exponential' or 'power'")


def _sample_pairs(xyz, maximum, rng):
    """Half local and half unrestricted pairs; memory is O(sites + maximum)."""
    n = len(xyz)
    if n < 2:
        return np.empty((0, 2), dtype=int)
    maximum = min(maximum, n * (n - 1) // 2)
    if n * (n - 1) // 2 <= maximum:
        return np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=int)
    pairs = set()
    tree = cKDTree(xyz)
    anchors = rng.integers(0, n, size=max(maximum, 16))
    _, neighbors = tree.query(xyz[anchors], k=min(9, n))
    for i, choices in zip(anchors, neighbors):
        # Coincident coordinates can mean that the first neighbor is not self.
        choices = np.asarray(choices)[np.asarray(choices) != i]
        if len(choices):
            j = int(rng.choice(choices))
            pairs.add((min(int(i), j), max(int(i), j)))
        if len(pairs) >= maximum // 2:
            break
    while len(pairs) < maximum:
        count = max(16, 2 * (maximum - len(pairs)))
        left = rng.integers(0, n, size=count)
        right = rng.integers(0, n - 1, size=count)
        right += right >= left
        for i, j in zip(left, right):
            pairs.add((min(int(i), int(j)), max(int(i), int(j))))
            if len(pairs) == maximum:
                break
    return np.asarray(sorted(pairs), dtype=int)


def _bivariate_normal_cdf(a, b, rho):
    """Plackett (1954) reduction identity with deterministic quadrature.

    See the full Biometrika reference in this module docstring. Quadrature is
    a numerical approximation to the bivariate-normal probability integral.
    """
    a, b, rho = np.broadcast_arrays(a, b, rho)
    t = rho[..., None] * (_QUAD_X + 1.0) / 2.0
    denom = 1.0 - t * t
    exponent = -(a[..., None] ** 2 - 2 * a[..., None] * b[..., None] * t
                 + b[..., None] ** 2) / (2.0 * denom)
    integral = rho / (4.0 * np.pi) * np.sum(
        _QUAD_W * np.exp(exponent) / np.sqrt(denom), axis=-1
    )
    return ndtr(a) * ndtr(b) + integral


def _binary_latent_correlation(p_left, p_right, p_both):
    a, b = ndtri(p_left), ndtri(p_right)
    lower = np.full_like(a, -0.999)
    upper = np.full_like(a, 0.999)
    for _ in range(42):
        middle = (lower + upper) / 2.0
        joint = _bivariate_normal_cdf(a, b, middle)
        lower = np.where(joint < p_both, middle, lower)
        upper = np.where(joint >= p_both, middle, upper)
    return (lower + upper) / 2.0


def _pearson(left, right):
    left, right = left - np.mean(left), right - np.mean(right)
    denom = np.sqrt(np.dot(left, left) * np.dot(right, right))
    return float(np.dot(left, right) / denom) if denom > 0 else np.nan


def fit_distance_model(values, lat, lon, kind="exponential", max_pairs=2000,
                       seed=42, transform="gaussian", min_overlap=30):
    """Fit one distance kernel from sampled pairs, without a dense covariance.

    Parameters
    ----------
    values : array (time, site)
        Standardized variable residuals, or rain occurrence / wet-day amounts.
        Mask dry-day amounts with NaN before calling. Pooling months fits a
        season-wide kernel; call separately by month for monthly kernels.
    transform : {'gaussian', 'binary', 'none'}
        Gaussian rank scores for continuous marginals; threshold inversion for
        0/1 occurrence; or direct Pearson correlations for existing Gaussian
        innovations. Binary inversion approximates simultaneous stationary
        occurrence, not the full Markov-chain joint law.
    min_overlap : int
        Required finite, contemporaneous observations for each sampled pair.

    Returns
    -------
    dict
        Directly accepted as ``GaussianSpatialField(model=result)``. Diagnostics
        include sampled pairs, physical/latent correlations, kernel residuals,
        and insufficient-data fallbacks. Negative correlations are retained in
        diagnostics but cannot be represented by these positive kernels.
    """
    values = np.asarray(values)
    _, _, xyz = _coordinates(lat, lon)
    if values.ndim != 2 or values.shape[1] != len(xyz) or any(n == 0 for n in values.shape):
        raise ValueError("values must have shape (time, site) matching coordinates")
    max_pairs = _integer(max_pairs, "max_pairs", 1)
    min_overlap = _integer(min_overlap, "min_overlap", 3)
    seed = _integer(seed, "seed")
    kernel_correlation(0, {"kind": kind})
    if transform not in {"gaussian", "binary", "none"}:
        raise ValueError("transform must be 'gaussian', 'binary', or 'none'")
    result = {"kind": kind, "range_km": 150.0}
    if kind == "power":
        result["alpha"] = 1.0

    # Scan narrow columns: no extra full time-by-site array or site-by-site array.
    valid_sites = []
    for start in range(0, values.shape[1], 256):
        block = np.asarray(values[:, start:start + 256], dtype=float)
        finite = np.isfinite(block)
        if transform == "binary" and np.any(finite & (block != 0) & (block != 1)):
            raise ValueError("transform='binary' requires occurrence values 0 or 1, with NaN for missing")
        count = finite.sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = np.nanmin(np.where(finite, block, np.nan), axis=0)
            hi = np.nanmax(np.where(finite, block, np.nan), axis=0)
        valid_sites.extend((start + np.flatnonzero((count >= min_overlap) & (hi > lo))).tolist())
    valid_sites = np.asarray(valid_sites, dtype=int)
    local_pairs = _sample_pairs(xyz[valid_sites], max_pairs, np.random.default_rng(_seed(seed, 31415)))
    candidates = valid_sites[local_pairs]
    ranks = {}
    if transform == "gaussian":
        for site in np.unique(candidates):
            column = np.asarray(values[:, site], dtype=float)
            good = np.isfinite(column)
            transformed = np.full(column.shape, np.nan, dtype=np.float32)
            transformed[good] = ndtri((rankdata(column[good], method="average") - 0.5) / good.sum())
            ranks[int(site)] = transformed

    pairs, raw_correlations, latent_correlations, counts, marginals = [], [], [], [], []
    for i, j in candidates:
        left, right = np.asarray(values[:, i], dtype=float), np.asarray(values[:, j], dtype=float)
        good = np.isfinite(left) & np.isfinite(right)
        count = int(good.sum())
        if count < min_overlap:
            continue
        left, right = left[good], right[good]
        raw = _pearson(left, right)
        if not np.isfinite(raw):
            continue
        if transform == "binary":
            if np.any((left != 0) & (left != 1)) or np.any((right != 0) & (right != 1)):
                raise ValueError("transform='binary' requires occurrence values 0 or 1, with NaN for missing")
            marginals.append((float(left.mean()), float(right.mean()), float(np.mean(left * right))))
            latent = np.nan  # inversion is vectorized after pairs have been gathered
        elif transform == "gaussian":
            latent = _pearson(ranks[int(i)][good].astype(float), ranks[int(j)][good].astype(float))
        else:
            latent = raw
        if transform != "binary" and not np.isfinite(latent):
            continue
        pairs.append((int(i), int(j)))
        counts.append(count)
        raw_correlations.append(raw)
        latent_correlations.append(latent)

    pair_count = len(pairs)
    result["pair_count"] = pair_count
    diagnostics = {
        "status": "fit" if pair_count else "fallback_no_usable_pairs",
        "transform": transform,
        "distance_metric": "Earth-sphere 3-D chord kilometres",
        "sampling": "approximately half nearest-neighbor and half unrestricted pairs",
        "n_valid_sites": len(valid_sites),
        "n_candidate_pairs": len(candidates),
        "min_overlap": min_overlap,
        "pairs": pairs,
        "n_overlap": counts,
        "physical_correlation": raw_correlations,
        "limitation": "Positive isotropic latent kernel; generated physical correlation is approximate."
    }
    result["diagnostics"] = diagnostics
    if not pair_count:
        diagnostics["fallback_reason"] = "No nonconstant site pairs with enough concurrent finite observations; range defaults to 150 km."
        return result

    pairs_array = np.asarray(pairs)
    distance = np.linalg.norm(xyz[pairs_array[:, 0]] - xyz[pairs_array[:, 1]], axis=1)
    physical = np.asarray(raw_correlations)
    if transform == "binary":
        p_left, p_right, p_both = np.asarray(marginals).T
        latent = _binary_latent_correlation(p_left, p_right, p_both)
    else:
        latent = np.asarray(latent_correlations)
    informative = (distance > 0) & (latent > 0.01) & (latent < 0.995)
    initial_range = float(np.median(-distance[informative] / np.log(latent[informative]))) if informative.any() else 150.0
    initial_range = np.clip(initial_range, 0.011, 999999.0)
    weights = np.sqrt(np.maximum(np.asarray(counts, dtype=float) - 3.0, 1.0))
    weights /= np.median(weights)

    def residual(parameters):
        model = {"kind": kind, "range_km": np.exp(parameters[0])}
        if kind == "power":
            model["alpha"] = np.exp(parameters[1])
        return (kernel_correlation(distance, model) - latent) * weights

    initial = [np.log(initial_range)] + ([0.0] if kind == "power" else [])
    lower = [np.log(0.01)] + ([np.log(0.05)] if kind == "power" else [])
    upper = [np.log(1e6)] + ([np.log(20.0)] if kind == "power" else [])
    fitted = least_squares(residual, initial, bounds=(lower, upper), loss="soft_l1", f_scale=0.1)
    result["range_km"] = float(np.exp(fitted.x[0]))
    if kind == "power":
        result["alpha"] = float(np.exp(fitted.x[1]))
    predicted_latent = kernel_correlation(distance, result)
    # Physical error is meaningful for binary inversion or identity transforms;
    # rank-Gaussian amount residuals need forward marginal simulation to estimate.
    if transform == "binary":
        predicted_physical = (_bivariate_normal_cdf(ndtri(p_left), ndtri(p_right), predicted_latent.clip(-0.999, 0.999))
                              - p_left * p_right) / np.sqrt(p_left * (1 - p_left) * p_right * (1 - p_right))
        binary_inversion_error = _bivariate_normal_cdf(ndtri(p_left), ndtri(p_right), latent) - p_both
        diagnostics["binary_joint_inversion_max_absolute_error"] = float(np.max(np.abs(binary_inversion_error)))
        diagnostics["physical_rmse"] = float(np.sqrt(np.mean((predicted_physical - physical) ** 2)))
        diagnostics["predicted_physical_correlation"] = predicted_physical.tolist()
    elif transform == "none":
        diagnostics["physical_rmse"] = float(np.sqrt(np.mean((predicted_latent - physical) ** 2)))
    else:
        diagnostics["physical_rmse"] = None
        diagnostics["physical_rmse_note"] = "Not inferred from latent ranks; evaluate generated physical fields."
    great_circle = 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(distance / (2 * EARTH_RADIUS_KM), 0, 1))
    diagnostics.update({
        "optimizer_success": bool(fitted.success),
        "distance_km": distance.tolist(),
        "latent_correlation": latent.tolist(),
        "predicted_latent_correlation": predicted_latent.tolist(),
        "latent_rmse": float(np.sqrt(np.mean((predicted_latent - latent) ** 2))),
        "negative_latent_fraction": float(np.mean(latent < 0)),
        "physical_latent_mean_absolute_difference": float(np.mean(np.abs(physical - latent))),
        "max_chord_shortfall_fraction": float(np.max(np.divide(great_circle - distance, great_circle, out=np.zeros_like(distance), where=great_circle > 0))),
        "parameters_at_bound": bool(np.any(np.isclose(fitted.x, lower, atol=1e-3)) or np.any(np.isclose(fitted.x, upper, atol=1e-3))),
    })
    return result


class DenseGaussianSpatialField:
    """Gaussian field with the specified finite-domain covariance.

    This reference backend factors the full kernel matrix by an eigendecomposition.
    It is useful for small station sets and for checking the Fourier approximation.
    It requires O(sites**2) memory and O(sites**3) setup; independent spatial tiles
    would change the joint law, so use ``GaussianSpatialField`` for tiled domains.
    Coincident coordinates are allowed and have perfectly correlated innovations.
    Exactness refers to the chosen Gaussian covariance up to roundoff, not to
    observed precipitation dependence or reproduction of Wilks (1998, 2002).
    """

    def __init__(self, lat, lon, model=None, seed=42, site_ids=None,
                 n_features=None, max_sites=512):
        self.lat, self.lon, xyz = _coordinates(lat, lon)
        self.seed = _integer(seed, "seed")
        self.model = dict(model or {"kind": "exponential", "range_km": 150.0})
        limit = _integer(max_sites, "max_sites", 1)
        if len(xyz) > limit:
            raise MemoryError(f"Dense spatial factorization is limited to {limit} sites; use the features backend")
        if site_ids is not None and len(site_ids) != len(xyz):
            raise ValueError("site_ids must match the coordinate length")
        distance = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        covariance = kernel_correlation(distance, self.model)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tolerance = 100 * np.finfo(float).eps * max(len(xyz), 1)
        if eigenvalues.min(initial=0.) < -tolerance:
            raise ValueError("The specified spatial covariance is not positive semidefinite")
        # Clip numerical roundoff only. Adding a nugget would change the model.
        self._factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.))[None, :]
        reconstructed = self._factor @ self._factor.T
        self.diagnostics = {
            "backend": "dense",
            "n_sites": len(xyz),
            "distance_metric": "Earth-sphere 3-D chord kilometres",
            "max_covariance_factorization_error": float(np.max(np.abs(reconstructed - covariance))),
            "minimum_eigenvalue": float(eigenvalues.min()),
            "tile_invariant": False,
        }

    def sample(self, n_members, step, stream):
        n_members = _integer(n_members, "n_members", 1)
        step, stream = _integer(step, "step", minimum=None), _integer(stream, "stream")
        rng = np.random.default_rng(_seed(self.seed, 16180, step, stream))
        return rng.normal(size=(n_members, len(self.lat))) @ self._factor.T


class GaussianSpatialField:
    """Shared Gaussian field with O(sites * n_features) stored features.

    ``n_features`` is the number of sampled frequencies; each contributes both
    sine and cosine. Gaussian coefficients make each site exactly N(0,1). Their
    *correlation* approximates the specified kernel, with Monte Carlo standard
    error bounded by ``1/sqrt(n_features)`` for a fixed pair. Feature frequencies
    and all draw coefficients depend on global seed/step/stream, never tile size
    or bounds. Instantiate tiles with identical model, seed, and n_features.

    Draws are reproducible for a fixed NumPy version; member prefixes are stable.
    ``site_ids`` is accepted for the common interface and checked for length but
    does not affect the spatial field, which is defined by coordinates.
    The paired Fourier map follows Rahimi and Recht (2007), Algorithm 1; the
    sphere/chord kernel is a package adaptation of the Wilks spatial mechanism.
    """

    def __init__(self, lat, lon, model=None, seed=42, n_features=256, site_ids=None):
        self.lat, self.lon, xyz = _coordinates(lat, lon)
        self.seed = _integer(seed, "seed")
        self.n_features = _integer(n_features, "n_features", 1)
        self.model = dict(model or {"kind": "exponential", "range_km": 150.0})
        kernel_correlation(0, self.model)
        if site_ids is not None and len(site_ids) != len(xyz):
            raise ValueError("site_ids must match the coordinate length")
        rng = np.random.default_rng(_seed(self.seed, 27182))
        # A 3-D isotropic Cauchy frequency vector is N(0,I)/abs(N(0,1)).
        frequencies = rng.normal(size=(self.n_features, 3))
        denominator = np.maximum(np.abs(rng.normal(size=self.n_features)), np.finfo(float).tiny)
        frequencies /= denominator[:, None]
        if self.model.get("kind", "exponential") == "power":
            rate = rng.gamma(float(self.model.get("alpha", 1.0)), scale=1.0 / self.model.get("range_km", 150.0), size=self.n_features)
        else:
            rate = np.full(self.n_features, 1.0 / self.model.get("range_km", 150.0))
        frequencies *= rate[:, None]
        angles = xyz @ frequencies.T
        norm = np.sqrt(self.n_features)
        self._features = np.concatenate((np.cos(angles), np.sin(angles)), axis=1) / norm
        self.diagnostics = {
            "n_frequencies": self.n_features,
            "n_sine_cosine_features": 2 * self.n_features,
            "marginal_variance": "exactly 1 (up to floating-point rounding)",
            "pair_correlation_standard_error_upper_bound": float(1.0 / norm),
            "distance_metric": "Earth-sphere 3-D chord kilometres",
            "tile_invariant": True,
        }

    def sample(self, n_members, step, stream):
        n_members = _integer(n_members, "n_members", 1)
        step, stream = _integer(step, "step", minimum=None), _integer(stream, "stream")
        rng = np.random.default_rng(_seed(self.seed, 16180, step, stream))
        coefficients = rng.normal(size=(n_members, 2 * self.n_features))
        return coefficients @ self._features.T

    def correlation_diagnostics(self, max_pairs=2000, seed=42):
        """Quantify this finite-feature covariance error on sampled pairs."""
        _, _, xyz = _coordinates(self.lat, self.lon)
        pairs = _sample_pairs(xyz, _integer(max_pairs, "max_pairs", 1), np.random.default_rng(_seed(_integer(seed, "seed"), 31415)))
        if not len(pairs):
            return {"pair_count": 0, "rmse": None, "max_absolute_error": None}
        distances = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
        approximate = np.sum(self._features[pairs[:, 0]] * self._features[pairs[:, 1]], axis=1)
        error = approximate - kernel_correlation(distances, self.model)
        return {"pair_count": len(pairs), "rmse": float(np.sqrt(np.mean(error ** 2))),
                "max_absolute_error": float(np.max(np.abs(error))),
                "minimum_feature_correlation": float(approximate.min())}


def _mix64(values):
    # Unsigned overflow is intentional: SplitMix64's bijective avalanche mixer.
    with np.errstate(over="ignore"):
        values = np.asarray(values, dtype=np.uint64)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))


class IndependentField:
    """Stateless independent site draws, invariant to tiles and member prefixes.

    Pass persistent unique global ``site_ids`` if sites are repeated or relocated.
    Otherwise IDs are hashes of normalized latitude/longitude, so an identical
    coordinate pair represents the same site. Unlike hashing with Python's
    ``hash()``, this calculation is stable across interpreter processes.
    """

    def __init__(self, lat, lon, model=None, seed=42, n_features=None, site_ids=None):
        self.lat, self.lon, _ = _coordinates(lat, lon)
        self.seed = _integer(seed, "seed")
        if site_ids is None:
            lat_bits = np.where(self.lat == 0, 0.0, self.lat).astype("<f8").view("<u8")
            lon_bits = np.where(self.lon == 0, 0.0, self.lon).astype("<f8").view("<u8")
            self._ids = _mix64(lat_bits) ^ _mix64(lon_bits ^ np.uint64(0x9E3779B97F4A7C15))
        else:
            ids = np.asarray(site_ids)
            if ids.ndim != 1 or len(ids) != len(self.lat):
                raise ValueError("site_ids must be one-dimensional and match coordinates")
            if len(set(map(str, ids))) != len(ids):
                raise ValueError("Explicit site_ids must be unique")
            self._ids = np.array([int.from_bytes(hashlib.blake2b(str(x).encode("utf-8"), digest_size=8).digest(), "little") for x in ids], dtype=np.uint64)
        self.diagnostics = {"tile_invariant": True, "kind": "independent"}

    def sample(self, n_members, step, stream):
        n_members = _integer(n_members, "n_members", 1)
        step, stream = _integer(step, "step", minimum=None), _integer(stream, "stream")
        nonce = _seed(self.seed, 14142, step, stream).generate_state(1, dtype=np.uint64)[0]
        member_ids = _mix64(np.arange(n_members, dtype=np.uint64) ^ np.uint64(0xD1B54A32D192ED03))
        counters = self._ids[None, :] ^ member_ids[:, None] ^ nonce
        hashed = _mix64(counters)
        # Use 52 bits, with midpoint quantiles strictly within (0, 1).
        uniform = ((hashed >> np.uint64(12)).astype(float) + 0.5) / float(1 << 52)
        return ndtri(uniform)


__all__ = ["fit_distance_model", "kernel_correlation", "GaussianSpatialField", "DenseGaussianSpatialField", "IndependentField"]
