"""Autograd training adaptations of the repository's three ID estimators.

Formulas/settings follow skdim.id.MLE (comb='mle'), TwoNN and CorrInt.
See src/scripts/id_estimation.py and the scikit-dimension estimator docs.
Correlation uses a sigmoid in log-distance space instead of hard counts.
These batch penalties are training objectives, not offline ID measurements.
"""
import math

import torch


METHODS = ("mle", "twonn", "corrint")


def _distinct_features(features):
    # Match the offline pipeline's removal of exact duplicate rows, preserving
    # first occurrence order. Only selection is detached, not the chosen rows.
    _, groups = torch.unique(features.detach(), dim=0, return_inverse=True)
    indices = torch.arange(len(features), device=features.device)
    first = torch.full((int(groups.max()) + 1,), len(features),
                       dtype=torch.long, device=features.device)
    first.scatter_reduce_(0, groups, indices, reduce="amin", include_self=True)
    return features[first.sort().values]


def id_penalty(features, method, mle_k=20, twonn_discard=0.1,
               corr_k1=10, corr_k2=20, corr_temperature=0.1, eps=1e-6):
    """Return scalar +log(D_batch) to encourage LOWER ID when minimized.

    The total training objective is CE + lambda * this penalty (lambda >= 0).
    The reference feature branch is detached.

    MLE: D is the harmonic mean of pointwise (k-1)/sum(log(r_k/r_j)).
    TwoNN: D is the zero-intercept empirical-CDF regression slope, trimming
      the largest 10% of r2/r1 ratios by default (same as the repo).
    CorrInt: D is a two-radius log correlation-integral slope; use sigmoid
      soft counts and differentiable median radii. This is an approximation.

    Neighbor counts shrink for small batches. Fewer than 3 distinct vectors
    receives the finite low-dimension penalty log(eps), with zero gradient.
    Floors/caps bound degenerate estimates; they can saturate gradients.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown ID loss: {method}")
    if features.ndim != 2 or len(features) == 0 or not features.is_floating_point():
        raise ValueError("Expected a nonempty floating-point feature matrix")
    if mle_k < 2 or not 0 <= twonn_discard < 1:
        raise ValueError("Require mle_k >= 2 and 0 <= twonn_discard < 1")
    if not 1 <= corr_k1 < corr_k2 or not math.isfinite(corr_temperature) or corr_temperature <= 0:
        raise ValueError("Require 1 <= corr_k1 < corr_k2 and finite positive temperature")
    if not 0 < eps < 1:
        raise ValueError("Require 0 < eps < 1")
    z = _distinct_features(features)
    if len(z) < 3:
        return features.sum() * 0 + math.log(eps)
    # Retain float64 for numerical tests; avoid half-precision distance logs.
    z = z if z.dtype == torch.float64 else z.float()
    distances = torch.cdist(z, z.detach(), compute_mode="donot_use_mm_for_euclid_dist")
    off_diagonal = ~torch.eye(len(z), device=z.device, dtype=torch.bool)
    scale = distances[off_diagonal].detach().mean().clamp_min(torch.finfo(z.dtype).tiny)
    distances = distances / scale
    neighbor_matrix = distances.masked_fill(~off_diagonal, float("inf"))

    if method == "mle":
        k = min(mle_k, len(z) - 1)
        radii = neighbor_matrix.topk(k, largest=False, sorted=True).values.clamp_min(eps)
        inverse_local = (radii[:, -1:].log() - radii[:, :-1].log()).mean(dim=1)
        # -log(mean(1/d_i)) = +log(harmonic_mean(d_i)).
        return -inverse_local.mean().clamp(eps, 1 / eps).log()

    if method == "twonn":
        radii = neighbor_matrix.topk(2, largest=False, sorted=True).values.clamp_min(eps)
        log_ratios = (radii[:, 1].log() - radii[:, 0].log()).sort().values
        retained = int(len(z) * (1 - twonn_discard))
        if retained < 2:
            return features.sum() * 0 + math.log(eps)
        x = log_ratios[:retained]
        empirical_cdf = torch.arange(retained, device=z.device, dtype=z.dtype) / len(z)
        y = -torch.log1p(-empirical_cdf)
        dimension = (x * y).sum() / x.square().sum().clamp_min(eps * eps)
    else:
        k2 = min(corr_k2, len(z) - 1)
        k1 = min(corr_k1, k2 - 1)
        radii = neighbor_matrix.topk(k2, largest=False, sorted=True).values
        # quantile interpolates the two middle values like numpy.median.
        r1 = torch.quantile(radii[:, k1 - 1], .5).clamp_min(eps)
        r2 = torch.quantile(radii[:, k2 - 1], .5).clamp_min(eps)
        log_pairs = distances[off_diagonal].clamp_min(eps).log()
        c1 = torch.sigmoid((r1.log() - log_pairs) / corr_temperature).mean()
        c2 = torch.sigmoid((r2.log() - log_pairs) / corr_temperature).mean()
        dimension = (c2.clamp_min(eps).log() - c1.clamp_min(eps).log()) / (
            r2.log() - r1.log()).clamp_min(eps)
    return dimension.clamp(eps, 1 / eps).log()
