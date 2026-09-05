import csv
import argparse
import numpy as np
import skdim

from pathlib import Path

from id_estimation import (
    run_id_scale_sweep,
    write_id_results,
    MLE_DEFAULT_N_NEIGHBORS,
    CORRINT_DEFAULT_K1,
    CORRINT_DEFAULT_K2,
    TWONN_DEFAULT_DISCARD_FRACTION
)


MANIFEST_FIELDS = ["name", "kind", "true_d", "ambient_dim", "n", "noise", "seed", "npy_path"]


def build_manifold_specs(n, seed, noise):
    specs = []

    # Bare d-sphere surfaces: ambient = d + 1, no padding, no noise --
    # the clean textbook case, so estimator behavior here is the
    # baseline everything else gets compared against.
    for d in (2, 5, 10):
        specs.append({
            "name": f"hypersphere_d{d}",
            "kind": "hypersphere",
            "true_d": d,
            "ambient_dim": d + 1,
            "n": n,
            "noise": 0.0,
            "seed": seed
        })

    # Linear/affine subspaces at ambient dimensions matching the VAE
    # sweep's own latent-size search space, so estimator bias here is
    # measured in the same high-ambient / low-intrinsic regime the real
    # embeddings actually live in.
    for d, ambient in ((5, 32), (9, 256), (20, 256)):
        specs.append({
            "name": f"affine_d{d}_amb{ambient}",
            "kind": "M9_Affine",
            "true_d": d,
            "ambient_dim": ambient,
            "n": n,
            "noise": noise,
            "seed": seed
        })

    # Curved (spherical) manifolds at the same (d, ambient) pairs as
    # the affine set above, so comparing the two separates "does
    # curvature confuse the estimator" from "does high ambient
    # dimension confuse the estimator" -- affine isolates the second on
    # its own, this adds the first on top of it.
    for d, ambient in ((5, 32), (9, 256), (20, 256)):
        specs.append({
            "name": f"curvedsphere_d{d}_amb{ambient}",
            "kind": "M1_Sphere",
            "true_d": d,
            "ambient_dim": ambient,
            "n": n,
            "noise": noise,
            "seed": seed
        })

    return specs


def generate_manifold(spec):
    if spec["kind"] == "hypersphere":
        return skdim.datasets.hyperSphere(
            n=spec["n"],
            d=spec["true_d"],
            random_state=spec["seed"]
        )

    # BenchmarkManifolds' noise_type only recognizes "uniform" or
    # "normal" -- "gaussian" silently fails to configure noise, so
    # "normal" is used deliberately here, not "gaussian".
    bm = skdim.datasets.BenchmarkManifolds(
        random_state=spec["seed"],
        noise_type="normal"
    )

    return bm.generate(
        name=spec["kind"],
        n=spec["n"],
        dim=spec["ambient_dim"],
        d=spec["true_d"],
        noise=spec["noise"]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic manifolds of known intrinsic dimension and run the ID-estimator scale sweep against each."
    )

    parser.add_argument("--n", type=int, default=20000, help="Points per synthetic manifold")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level for the affine/curved manifolds (bare hyperspheres are always noise-free)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for manifold generation and subsampling")
    parser.add_argument("--mock-data-dir", type=str, default="mock_data", help="Directory the generated point clouds and manifest are written to")
    parser.add_argument("--results-dir", type=str, default="results/id_estimation_synthetic", help="Directory the per-manifold scale-sweep CSVs are written to")
    parser.add_argument("--size-fractions", type=str, default="0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--num-repeats", type=int, default=5)
    parser.add_argument("--corrint-max-n", type=int, default=10000)

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

    return parser.parse_args()


def main():
    args = parse_args()

    mock_data_dir = Path(args.mock_data_dir)
    mock_data_dir.mkdir(parents=True, exist_ok=True)

    size_fractions = tuple(float(x) for x in args.size_fractions.split(","))
    specs = build_manifold_specs(args.n, args.seed, args.noise)

    manifest_rows = []

    print(
        f"Neighbor settings: MLE n_neighbors={args.mle_n_neighbors}, "
        f"CorrInt k1={args.corrint_k1}/k2={args.corrint_k2}, "
        f"Two-NN discard_fraction={args.twonn_discard_fraction}\n"
    )

    for spec in specs:
        print(f"\n=== Generating {spec['name']} (true d={spec['true_d']}, ambient={spec['ambient_dim']}) ===")

        points = generate_manifold(spec)

        npy_path = mock_data_dir / f"{spec['name']}.npy"
        np.save(npy_path, points)
        print(f"Saved {points.shape[0]} points x {points.shape[1]} dims to {npy_path}")

        manifest_rows.append({**spec, "npy_path": str(npy_path)})

        records = run_id_scale_sweep(
            points,
            size_fractions=size_fractions,
            num_repeats=args.num_repeats,
            corrint_max_n=args.corrint_max_n,
            random_state=args.seed,
            label=spec["name"],
            true_d=spec["true_d"],
            mle_n_neighbors=args.mle_n_neighbors,
            corrint_k1=args.corrint_k1,
            corrint_k2=args.corrint_k2,
            twonn_discard_fraction=args.twonn_discard_fraction
        )

        write_id_results(records, Path(args.results_dir) / f"{spec['name']}.csv")

    manifest_path = mock_data_dir / "manifest.csv"

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nWrote manifest for {len(manifest_rows)} manifolds to {manifest_path}")


if __name__ == "__main__":
    main()
