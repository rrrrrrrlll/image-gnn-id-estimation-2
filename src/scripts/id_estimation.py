import csv
import numpy as np
import skdim

from pathlib import Path
from datetime import datetime


# skdim.id.MLE's own class-level default neighborhood size, used by
# fit_transform() whenever n_neighbors isn't passed explicitly and
# neighborhood_based=True (the default mode, and the only mode this
# project uses). This is NOT the same as the constructor's K=5
# parameter -- K is only consulted in the non-default
# neighborhood_based=False mode, so leaving n_neighbors unset means
# MLE silently estimates from 20 neighbors regardless of K. Passing it
# explicitly here just makes that choice visible and tunable instead
# of hidden inside the library.
MLE_DEFAULT_N_NEIGHBORS = 20

# CorrInt's neighbor-rank indices, converted into the two radii its
# correlation-integral slope is fit between. These are the library's
# own defaults (Levina & Bickel 2005's general-purpose averaging
# range), not a fitted optimum -- shrinking corrint_k2 pulls the fit
# window toward smaller local radii, which is the theoretically
# correct way to reduce curvature bias for this method.
CORRINT_DEFAULT_K1 = 10
CORRINT_DEFAULT_K2 = 20

# Two-NN has no neighborhood-size knob at all -- it is fixed to
# exactly 2 nearest neighbors by definition (that minimal neighborhood
# is the whole premise of Facco et al. 2019's method). discard_fraction
# only trims outlier points by their N2/N1 ratio before the fit; it
# does not change curvature bias the way the parameters above do.
TWONN_DEFAULT_DISCARD_FRACTION = 0.1

_KNOWN_METHODS = ("mle", "twonn", "corrint")


def _fit_one(method, sample, mle_n_neighbors, corrint_k1, corrint_k2, twonn_discard_fraction):
    """
    Builds and fits a single estimator, returning (estimated_id,
    neighbor_param) where neighbor_param is a short human-readable
    string recording exactly what neighborhood setting produced this
    estimate -- so that setting is always visible in the output CSV
    instead of being an invisible library default the way MLE's k=20
    used to be.
    """
    if method == "mle":
        estimator = skdim.id.MLE()
        estimated_id = float(estimator.fit_transform(sample, n_neighbors=mle_n_neighbors))
        neighbor_param = f"n_neighbors={mle_n_neighbors}"
    elif method == "twonn":
        estimator = skdim.id.TwoNN(discard_fraction=twonn_discard_fraction)
        estimated_id = float(estimator.fit_transform(sample))
        neighbor_param = f"discard_fraction={twonn_discard_fraction}"
    elif method == "corrint":
        estimator = skdim.id.CorrInt(k1=corrint_k1, k2=corrint_k2)
        estimated_id = float(estimator.fit_transform(sample))
        neighbor_param = f"k1={corrint_k1},k2={corrint_k2}"
    else:
        raise ValueError(f"Unknown ID estimation method: {method}")

    return estimated_id, neighbor_param


def run_id_scale_sweep(
    points,
    size_fractions=(0.05, 0.1, 0.25, 0.5, 1.0),
    num_repeats=5,
    methods=("mle", "twonn", "corrint"),
    corrint_max_n=10000,
    random_state=None,
    label=None,
    true_d=None,
    mle_n_neighbors=MLE_DEFAULT_N_NEIGHBORS,
    corrint_k1=CORRINT_DEFAULT_K1,
    corrint_k2=CORRINT_DEFAULT_K2,
    twonn_discard_fraction=TWONN_DEFAULT_DISCARD_FRACTION
):
    """
    Runs each requested ID estimator across several sample sizes
    (fractions of the available points), repeating each size several
    times with an independent random subsample. Instead of one number
    per estimator, this gives a curve: does the estimate keep moving as
    n grows, or has it already settled by the sizes the real data
    actually has? Every estimator sees the SAME subsample at a given
    (fraction, repeat) -- except CorrInt, which is additionally capped
    at corrint_max_n points drawn from that same subsample, since its
    pairwise-distance computation does not scale the way the two k-NN
    based estimators do. The sample size actually used is recorded per
    row for every method, so the cap is never silently hidden the way
    it was in the original single-run script.

    points: (N, F) array -- ambient-space coordinates, any label column
        already stripped.
    label: optional string identifying what this is (dataset name, or
        a synthetic manifold name) -- stored in every output row so
        results from different calls can be told apart once combined.
    true_d: optional known ground-truth intrinsic dimension. Only
        meaningful for synthetic manifolds -- leave None for real data.
        When given, each row also records the signed and absolute
        error against it.
    mle_n_neighbors: neighborhood size MLE estimates each point's local
        dimension from (passed straight to fit_transform). Defaults to
        the library's own silent default of 20; pass a smaller value
        (e.g. 10) to trade curvature bias for more variance, per
        Campadelli et al. 2015's bias-variance discussion of this
        estimator.
    corrint_k1 / corrint_k2: neighbor-rank indices CorrInt is
        constructed with. Shrinking corrint_k2 tightens the
        correlation-integral fit toward smaller local radii, reducing
        curvature bias at the cost of a noisier fit.
    twonn_discard_fraction: outlier-trim fraction Two-NN is constructed
        with. Has no effect on curvature bias -- Two-NN's neighborhood
        size is fixed at 2 by definition and cannot be tuned.

    Returns a list of record dicts, one per (fraction, repeat, method).
    """
    rng = np.random.default_rng(random_state)
    n_total = points.shape[0]

    records = []

    for fraction in size_fractions:
        n_target = max(2, min(n_total, round(fraction * n_total)))
        is_full = n_target >= n_total
        repeats = 1 if is_full else num_repeats

        for repeat in range(repeats):
            if is_full:
                idx = np.arange(n_total)
            else:
                idx = rng.choice(n_total, size=n_target, replace=False)

            sample = points[idx]

            for method in methods:
                if method not in _KNOWN_METHODS:
                    raise ValueError(f"Unknown ID estimation method: {method}")

                method_sample = sample
                n_used = n_target

                if method == "corrint" and n_target > corrint_max_n:
                    sub_idx = rng.choice(n_target, size=corrint_max_n, replace=False)
                    method_sample = sample[sub_idx]
                    n_used = corrint_max_n

                record = {
                    "label": label,
                    "true_d": true_d,
                    "method": method,
                    "fraction": fraction,
                    "n_target": n_target,
                    "n_used": n_used,
                    "repeat": repeat,
                    "neighbor_param": None,
                    "estimated_id": None,
                    "signed_error": None,
                    "abs_error": None,
                    "status": "ok",
                    "error_message": "",
                    "timestamp": datetime.now().isoformat()
                }

                try:
                    estimated_id, neighbor_param = _fit_one(
                        method,
                        method_sample,
                        mle_n_neighbors,
                        corrint_k1,
                        corrint_k2,
                        twonn_discard_fraction
                    )
                    record["estimated_id"] = estimated_id
                    record["neighbor_param"] = neighbor_param

                    if true_d is not None:
                        record["signed_error"] = estimated_id - true_d
                        record["abs_error"] = abs(estimated_id - true_d)
                except Exception as e:
                    record["status"] = "failed"
                    record["error_message"] = str(e)

                records.append(record)

                print(
                    f"[{label}] {method:>7s} | fraction={fraction:>5} | "
                    f"n_used={n_used:>6d} | repeat={repeat} | "
                    f"estimate={record['estimated_id']} | status={record['status']}"
                )

    return records


def write_id_results(records, results_path):
    if not records:
        return None

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(records[0].keys())
    write_header = not results_path.exists()

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerows(records)

    print(f"Wrote {len(records)} row(s) to {results_path}")

    return results_path
