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

# Public so gen_graph_id.py can validate --methods before running anything,
# instead of only failing partway through the first fraction's sweep.
KNOWN_METHODS = ("mle", "twonn", "corrint")


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
        # fit_transform() must be passed comb="mle" explicitly -- MLE.fit()'s own
        # default is comb="mle" (the correct Levina-Bickel harmonic-mean
        # combination of pointwise estimates), but MLE does not override
        # fit_transform(), so it inherits LocalEstimator.fit_transform()'s own
        # default of comb="mean" (a plain arithmetic mean) and silently passes
        # that into fit() instead, overriding MLE's better default. Confirmed by
        # running the actual skdim source: the arithmetic mean consistently
        # overestimates relative to the harmonic mean, by roughly 5% of the
        # estimate and growing in absolute terms as the dimension itself grows.
        estimated_id = float(estimator.fit_transform(sample, n_neighbors=mle_n_neighbors, comb="mle"))
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


def deduplicate_embeddings(points):
    """
    Drop exact-duplicate rows before any ID estimation runs on them.

    Duplicate embeddings (a repeated source image, or distinct images the
    VAE happened to encode to the same latent point) are not informative
    for a manifold-based ID estimate -- and, more importantly, MLE's
    comb="mle" harmonic-mean aggregation is not robust to even one of
    them: a point whose nearest neighbor sits at distance exactly 0 gets a
    per-point local-dimension estimate of exactly 0 (log(Rk / 0) is +inf,
    which dominates that point's sum), and a single 0 in a harmonic mean
    (1 / mean(1 / estimates)) collapses the WHOLE aggregate to 0.0, no
    matter how many thousands of well-behaved points are also in the
    sample. FER2013 (~7.9% exact-duplicate rows) and CelebA (~0.1%) both
    hit this in practice once comb was fixed from its previous silent
    "mean" default to the correct "mle" default -- see
    results/id_estimation for the before/after. Deduplicating once, up
    front, removes the failure mode at its source rather than working
    around it downstream, and (as a side benefit) means every size
    fraction is a fraction of the genuinely distinct points available,
    not padded out by copies.

    points: (N, F) array -- ambient-space coordinates, any label column
        already stripped.

    Returns (deduplicated_points, n_dropped).
    """
    # View each row as a single structured-dtype element so np.unique can
    # compare whole rows at once instead of column-by-column; this is the
    # same trick used to survey duplicate rates during the audit, so the
    # count this reports matches what was already found empirically.
    contiguous = np.ascontiguousarray(points)
    structured = contiguous.view(
        [("", contiguous.dtype)] * contiguous.shape[1]
    )
    _, first_occurrence = np.unique(structured, return_index=True)
    # np.unique sorts its output; re-sort the kept indices back into their
    # original order so row order (and thus anything keyed off position)
    # stays as close to the input as deduplication allows.
    keep_idx = np.sort(first_occurrence)
    n_dropped = points.shape[0] - len(keep_idx)
    return points[keep_idx], n_dropped


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
                if method not in KNOWN_METHODS:
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


def write_id_results(records, results_dir, label):
    """
    Writes records into one CSV per method, under
    <results_dir>/<method>/<label>.csv -- e.g.
    results/id_estimation/mle/mnist.csv,
    results/id_estimation/twonn/mnist.csv,
    results/id_estimation/corrint/mnist.csv, instead of one file mixing all
    three methods together. This means a --methods corrint-only rerun (see
    gen_graph_id.py) only ever touches results/id_estimation/corrint/<label>.csv
    -- MLE's and Two-NN's own files, and their history, are untouched.

    Each per-method file is still append-only, same as before: an existing
    file gets new rows appended (no header rewritten), a missing one gets
    created with a header. Re-running the same (dataset, method) is still the
    caller's responsibility to archive/clear first, exactly as before.

    records: list of record dicts, as returned by run_id_scale_sweep() --
        every record must have a "method" key naming which file it goes to.
    results_dir: the shared parent directory (e.g. "results/id_estimation");
        method subfolders are created under it as needed.
    label: dataset/label name, used as the CSV's filename stem.

    Returns {method: path_written} for whichever methods were present in
    records (empty dict if records was empty).
    """
    if not records:
        return {}

    results_dir = Path(results_dir)

    by_method = {}
    for record in records:
        by_method.setdefault(record["method"], []).append(record)

    written = {}
    for method, method_records in by_method.items():
        method_path = results_dir / method / f"{label}.csv"
        method_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(method_records[0].keys())
        write_header = not method_path.exists()

        with open(method_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            writer.writerows(method_records)

        print(f"Wrote {len(method_records)} row(s) to {method_path}")
        written[method] = method_path

    return written
