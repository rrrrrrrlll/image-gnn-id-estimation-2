import re
import argparse
import numpy as np

from pathlib import Path

from id_estimation import (
    run_id_scale_sweep,
    write_id_results,
    deduplicate_embeddings,
    KNOWN_METHODS,
    MLE_DEFAULT_N_NEIGHBORS,
    CORRINT_DEFAULT_K1,
    CORRINT_DEFAULT_K2,
    TWONN_DEFAULT_DISCARD_FRACTION
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate Intrinsic Dimension of saved embeddings across several sample sizes."
    )

    parser.add_argument(
        "-e",
        "--embedding_path",
        type=str,
        help="Path to the saved .npy embeddings file (e.g., pathmnist_train_embeddings.npy)",
        required=True
    )

    parser.add_argument(
        "--size-fractions",
        type=str,
        default="0.05,0.1,0.25,0.5,1.0",
        help="Comma-separated fractions of the available points to sample at (last value is treated as the full-sample headline result)"
    )

    parser.add_argument(
        "--methods",
        type=str,
        default="mle,twonn,corrint",
        help=(
            "Comma-separated subset of {mle,twonn,corrint} to run, e.g. "
            "--methods corrint to regenerate just CorrInt's results/id_estimation/corrint/<label>.csv "
            "without touching MLE's or Two-NN's own files."
        )
    )

    parser.add_argument(
        "--num-repeats",
        type=int,
        default=5,
        help="Independent random subsamples per size below 100%% (the full-size point always runs once)"
    )

    parser.add_argument(
        "--corrint-max-n",
        type=int,
        default=100000,
        help="Cap on the sample size handed to CorrInt specifically, since its pairwise-distance cost does not scale like the k-NN based estimators (raised 10000 -> 25000 -> 50000 -> 100000 as more compute became available; the 25000 run took ~4 minutes for all 7 datasets)"
    )

    parser.add_argument(
        "--mle-n-neighbors",
        type=int,
        default=MLE_DEFAULT_N_NEIGHBORS,
        help="Neighborhood size MLE estimates each point's local dimension from. Smaller values reduce curvature bias at the cost of more variance (default matches skdim's own silent default of 20)"
    )

    parser.add_argument(
        "--corrint-k1",
        type=int,
        default=CORRINT_DEFAULT_K1,
        help="Lower neighbor-rank index CorrInt's correlation-integral fit starts from"
    )

    parser.add_argument(
        "--corrint-k2",
        type=int,
        default=CORRINT_DEFAULT_K2,
        help="Upper neighbor-rank index CorrInt's correlation-integral fit ends at. Shrinking this tightens the fit toward smaller local radii, reducing curvature bias"
    )

    parser.add_argument(
        "--twonn-discard-fraction",
        type=float,
        default=TWONN_DEFAULT_DISCARD_FRACTION,
        help="Fraction of points with the largest N2/N1 ratio dropped before Two-NN's fit. Two-NN's neighborhood size is fixed at 2 by definition and cannot itself be tuned"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/id_estimation",
        help="Directory the per-dataset scale-sweep CSV is written to"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subsampling"
    )

    return parser.parse_args()


def extract_label(embedding_path):
    stem = Path(embedding_path).stem
    match = re.match(r"(.+?)_(train|test)_embeddings$", stem)
    return match.group(1) if match else stem


def main():
    args = parse_args()

    print(f"Loading embeddings from: {args.embedding_path}")
    try:
        data = np.load(Path(args.embedding_path))
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # gen_graph_embedding.py appends the target label as the last column.
    # We must strip the labels off to only analyze the latent space features.
    embeddings = data[:, :-1]

    # Drop exact-duplicate embedding rows before anything is subsampled or
    # estimated from them -- see deduplicate_embeddings() for why this
    # matters beyond just avoiding redundant points: MLE's comb="mle"
    # aggregation collapses to exactly 0.0 if even one duplicate survives
    # into a sample (confirmed on FER2013 and CelebA after the comb fix).
    n_before_dedup = embeddings.shape[0]
    embeddings, n_dropped = deduplicate_embeddings(embeddings)
    if n_dropped:
        print(
            f"Dropped {n_dropped} exact-duplicate embedding row(s) "
            f"({100 * n_dropped / n_before_dedup:.3f}% of {n_before_dedup}) before ID estimation."
        )

    print(f"Data loaded successfully. Shape (Samples, Features): {embeddings.shape}")

    label = extract_label(args.embedding_path)
    size_fractions = tuple(float(x) for x in args.size_fractions.split(","))

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    unknown_methods = [m for m in methods if m not in KNOWN_METHODS]
    if unknown_methods:
        print(f"Error: unknown method(s) {unknown_methods} -- choose from {KNOWN_METHODS}")
        return

    print(f"Running {', '.join(methods)} across sample-size fractions: {size_fractions}")
    print(
        f"Neighbor settings: MLE n_neighbors={args.mle_n_neighbors}, "
        f"CorrInt k1={args.corrint_k1}/k2={args.corrint_k2}, "
        f"Two-NN discard_fraction={args.twonn_discard_fraction}\n"
    )

    records = run_id_scale_sweep(
        embeddings,
        size_fractions=size_fractions,
        num_repeats=args.num_repeats,
        methods=methods,
        corrint_max_n=args.corrint_max_n,
        random_state=args.seed,
        label=label,
        mle_n_neighbors=args.mle_n_neighbors,
        corrint_k1=args.corrint_k1,
        corrint_k2=args.corrint_k2,
        twonn_discard_fraction=args.twonn_discard_fraction
    )

    written_paths = write_id_results(records, args.results_dir, label)

    full_fraction = size_fractions[-1]
    full_rows = [r for r in records if r["fraction"] == full_fraction and r["status"] == "ok"]

    print("\n==============================")
    print(f"Full-sample ({full_fraction * 100:.0f}%) estimates for {label}")
    print("==============================")
    for r in full_rows:
        print(f"{r['method']:>7s} Estimated ID: {r['estimated_id']:.4f}  (n={r['n_used']})")

    print("\nFull scale-sweep (all fractions, all repeats) written to:")
    for method in sorted(written_paths):
        print(f"  {method:>7s} -> {written_paths[method]}")


if __name__ == "__main__":
    main()
