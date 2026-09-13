import sys
import yaml
import argparse

from pathlib import Path

from training.train import Trainer


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset_config", type=str, help="Training configuration file", required=True)
    parser.add_argument("-m", "--model_config", type=str, help="Dataset setup configuration file", required=True)
    parser.add_argument("-t", "--training_config", type=str, help="Model configuration file", required=True)
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/mlp_baseline",
        help="Directory the per-dataset MLP-baseline CSV is written to"
    )
    parser.add_argument(
        "--n-target",
        type=int,
        default=None,
        help=(
            "If given, each restart trains on a random n-target-row subsample of "
            "the training set (capped at however many rows this dataset actually "
            "has) instead of the full set -- for a fair comparison against a GNN "
            "gap-curve point trained at the same matched size. Default (None) "
            "trains on the full training set, matching the original behavior."
        )
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of independent (subsample draw + model init) restarts, appended as separate rows."
    )

    return parser.parse_args(args)


def main(sys_args):
    args = parse_args(sys_args)
    config_paths = {"dataset_config": args.dataset_config, "training_config": args.training_config, "model_config": args.model_config}
    config = {}
    for config_name, config_file_path in config_paths.items():
        with open(Path(config_file_path), "r") as f:
            config[config_name] = yaml.safe_load(f)

    trainer = Trainer(**config)

    # Trainer.dataset_name is derived from dataset_class, which is the
    # generic FlatEmbeddingDataset for every entry under config/mlp/ --
    # override it with the actual per-dataset config filename so results
    # land in a per-dataset file instead of colliding into one.
    trainer.dataset_name = Path(args.dataset_config).stem

    trainer.train_eval_mlp_baseline(
        results_dir=args.results_dir,
        n_target=args.n_target,
        num_seeds=args.num_seeds
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
