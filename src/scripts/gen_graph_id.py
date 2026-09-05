import re
import argparse
import numpy as np

from pathlib import Path

from id_estimation import (
    run_id_scale_sweep,
    write_id_results,
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
        "--num-repeats",
        type=int,
        default=5,
        help="Independent random subsamples per size below 100%% (the full-size point always runs once)"
    )

    parser.add_argument(
        "--corrint-max-n",
        type=int,
        default=10000,
        help="Cap on the sample size handed to CorrInt specifically, since its pairwise-distance cost does not scale like the k-NN based estimators"
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

    print(f"Data loaded successfully. Shape (Samples, Features): {embeddings.shape}")

    label = extract_label(args.embedding_path)
    size_fractions = tuple(float(x) for x in args.size_fractions.split(","))

    print(f"Running MLE / Two-NN / CorrInt across sample-size fractions: {size_fractions}")
    print(
        f"Neighbor settings: MLE n_neighbors={args.mle_n_neighbors}, "
        f"CorrInt k1={args.corrint_k1}/k2={args.corrint_k2}, "
        f"Two-NN discard_fraction={args.twonn_discard_fraction}\n"
    )

    records = run_id_scale_sweep(
        embeddings,
        size_fractions=size_fractions,
        num_repeats=args.num_repeats,
        corrint_max_n=args.corrint_max_n,
        random_state=args.seed,
        label=label,
        mle_n_neighbors=args.mle_n_neighbors,
        corrint_k1=args.corrint_k1,
        corrint_k2=args.corrint_k2,
        twonn_discard_fraction=args.twonn_discard_fraction
    )

    results_path = Path(args.results_dir) / f"{label}.csv"
    write_id_results(records, results_path)

    full_fraction = size_fractions[-1]
    full_rows = [r for r in records if r["fraction"] == full_fraction and r["status"] == "ok"]

    print("\n==============================")
    print(f"Full-sample ({full_fraction * 100:.0f}%) estimates for {label}")
    print("==============================")
    for r in full_rows:
        print(f"{r['method']:>7s} Estimated ID: {r['estimated_id']:.4f}  (n={r['n_used']})")

    print(f"\nFull scale-sweep (all fractions, all repeats) written to: {results_path}")


if __name__ == "__main__":
    main()
