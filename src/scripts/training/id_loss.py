"""Differentiable method-of-moments local ID (LDReg adaptation).

Reference: https://github.com/HanxunH/LDReg (ICLR 2024).
Neighbors are selected in each seed batch; the reference branch is detached.
"""
import torch


def local_log_id(features, k=20, eps=1e-6):
    if features.ndim != 2 or features.shape[0] < 3:
        raise ValueError("Local ID needs at least three feature vectors")
    if k < 2 or eps <= 0:
        raise ValueError("Require k >= 2 and eps > 0")
    k = min(k, features.shape[0] - 1)
    # Exclude the actual self index, including when other rows are duplicates.
    distances = torch.cdist(features.float(), features.detach().float(),
                            compute_mode="donot_use_mm_for_euclid_dist")
    diagonal = torch.eye(len(features), dtype=torch.bool, device=features.device)
    radii = distances.masked_fill(diagonal, float("inf")).topk(
        k, largest=False, sorted=True).values
    # Mean of k-1 inner distances divided by the kth radius minus that mean.
    # Relative ratios avoid dependence on feature scale. A collapsed batch gets
    # the lower clamp (small ID), rather than an artificially high dimension.
    ratio = (radii[:, :-1] / radii[:, -1:].clamp_min(eps)).mean(dim=1)
    ratio = ratio.clamp(eps, 1 - eps)
    return torch.log(ratio) - torch.log1p(-ratio)
