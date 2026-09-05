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
        results_dir=args.results_dir
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
