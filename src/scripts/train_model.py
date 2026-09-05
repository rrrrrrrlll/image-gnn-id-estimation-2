import sys
import yaml
import wandb
import argparse
import os
import torch

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

    parser.add_argument(
        "--sweep_id",
        type=str,
        required=True,
        help="W&B sweep ID this agent should pull trials from (one sweep per dataset)"
    )

    parser.add_argument(
        "--project",
        type=str,
        default="image-gnn",
        help="W&B project name for the sweep"
    )
 
    return parser.parse_args(args)

def main():
    wandb.init()
    args = parse_args(sys.argv[1:])

    config_paths = {
        "dataset_config": args.dataset_config,
        "training_config": args.training_config,
        "model_config": args.model_config
    }

    config = {}
    for config_name, config_file_path in config_paths.items():
        with open(Path(config_file_path), "r") as f:
            config[config_name] = yaml.safe_load(f)

    sweep_args = {
        k: v
        for k, v in wandb.config.items()
        if k != "in_channels"
    }

    config["model_config"]["args"].update(sweep_args)

    trainer = Trainer(**config)
    trainer.train()

if __name__ == "__main__":
    cli_args = parse_args(sys.argv[1:])

    wandb.agent(
        sweep_id=cli_args.sweep_id,
        function=main,
        project=cli_args.project
    )