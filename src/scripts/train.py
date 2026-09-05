import sys
import yaml
import argparse

from pathlib import Path

from training.train import Trainer

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", 
        "--dataset_config", 
        type=str, 
        help="Training configuration file", 
        required=True
    )

    parser.add_argument(
        "-m", 
        "--model_config", 
        type=str, 
        help="Dataset setup configuration file", 
        required=True
    )

    parser.add_argument(
        "-t",
        "--training_config", 
        type=str, 
        help="Model configuration file", 
        required=True
    )
 
    return parser.parse_args(args)

def main(sys_args):
    args = parse_args(sys_args)

    config_paths = {
        "dataset_config": args.dataset_config,
        "training_config": args.training_config,
        "model_config": args.model_config
    }

    config = {}
    for config_name, config_file_path in config_paths.items():
        with open(Path(config_file_path), "r") as f:
            config[config_name] = yaml.safe_load(f)

    trainer = Trainer(**config)

    # Trainer.dataset_name is derived from dataset_class, which is the
    # generic GraphEmbeddedDataset for every entry under config/gnn/ --
    # override it with the actual per-dataset config filename so that if
    # save_model is ever turned on (the README's own recommended
    # real-experiment setting), checkpoints land in a per-dataset folder
    # instead of every dataset's checkpoints colliding into
    # src/scripts/checkpoints/graphembedded/.
    trainer.dataset_name = Path(args.dataset_config).stem

    trainer.train_eval_grow_graph()

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))