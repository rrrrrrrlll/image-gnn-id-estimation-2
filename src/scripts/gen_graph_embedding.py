import sys
import yaml
import argparse

import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path

from models.models import CNNVAEModel
from data_preproc.datasets import build_datasets
from torch_geometric.loader import DataLoader


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset_config",
        type=str,
        required=True,
        help="Dataset configuration YAML file"
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="Trained CNN-VAE checkpoint"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )

    return parser.parse_args(args)


def targets_to_numpy(targets):
    if isinstance(targets, torch.Tensor):
        return targets.detach().cpu().numpy()

    return np.asarray(targets)


def main(sys_args):
    args = parse_args(sys_args)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------------------
    # 1. Load dataset config
    # --------------------------------------------------

    with open(Path(args.dataset_config), "r") as f:
        dataset_config = yaml.safe_load(f)

    dataset_class = dataset_config["train"]["dataset_class"]

    # MNISTDataset -> MNIST
    # FMNISTDataset -> FMNIST
    # PathMNISTDataset -> PathMNIST
    dataset_label = dataset_class.replace("Dataset", "")
    dataset_name = dataset_label.lower()

    print(f"Dataset: {dataset_label}")

    # --------------------------------------------------
    # 2. Load checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    print("Checkpoint dataset:", checkpoint["dataset_name"])
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Checkpoint test loss:", checkpoint["test_loss"])
    print("Checkpoint model config:")
    print(checkpoint["model_config"])

    # Prevent accidentally using e.g. MNIST checkpoint
    # with PathMNIST data.
    checkpoint_dataset = checkpoint.get("dataset_name")

    if (
        checkpoint_dataset is not None
        and checkpoint_dataset.lower() != dataset_name
    ):
        raise ValueError(
            f"Dataset mismatch: config is {dataset_name}, "
            f"but checkpoint is {checkpoint_dataset}"
        )

    # --------------------------------------------------
    # 3. Reconstruct model automatically
    # --------------------------------------------------

    model_config = checkpoint["model_config"]

    if model_config["model"] != "CNNVAEModel":
        raise ValueError(
            "This embedding script currently expects "
            "a CNNVAEModel checkpoint."
        )

    model_args = model_config["args"].copy()

    # pre_init creates `blocks` from
    # block_1, block_2, block_3 when needed.
    model_args = CNNVAEModel.pre_init(model_args)

    model = CNNVAEModel(**model_args)

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model = model.to(device)
    model.eval()

    print("\nLoaded model:")
    print(model)

    print(
        "\nLatent dimension:",
        model.latent_size
    )

    # --------------------------------------------------
    # 4. Load datasets
    # --------------------------------------------------

    train_ds, test_ds = build_datasets(
        dataset_config
    )

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(
        f"\nTrain samples: {len(train_ds)}"
    )

    print(
        f"Test samples: {len(test_ds)}"
    )

    # --------------------------------------------------
    # 5. Generate latent mean embeddings
    # --------------------------------------------------

    all_train_mu = []
    all_test_mu = []

    with torch.no_grad():

        print("\nGenerating train embeddings...")

        for train_batch in tqdm(train_dl):

            train_batch = train_batch.to(device)

            # We only need the latent mean.
            # No need to run the decoder.
            mu, logvar = model.encoder(
                train_batch.x
            )

            all_train_mu.append(
                mu.detach().cpu().numpy()
            )

        print("\nGenerating test embeddings...")

        for test_batch in tqdm(test_dl):

            test_batch = test_batch.to(device)

            mu, logvar = model.encoder(
                test_batch.x
            )

            all_test_mu.append(
                mu.detach().cpu().numpy()
            )

    train_embeddings = np.concatenate(
        all_train_mu,
        axis=0
    )

    test_embeddings = np.concatenate(
        all_test_mu,
        axis=0
    )

    # --------------------------------------------------
    # 6. Add class labels
    # --------------------------------------------------

    train_targets = targets_to_numpy(
        train_ds.targets
    ).reshape(-1, 1)

    test_targets = targets_to_numpy(
        test_ds.targets
    ).reshape(-1, 1)

    train_output = np.concatenate(
        [
            train_embeddings,
            train_targets
        ],
        axis=1
    )

    test_output = np.concatenate(
        [
            test_embeddings,
            test_targets
        ],
        axis=1
    )

    # --------------------------------------------------
    # 7. Automatically create dataset output directory
    # --------------------------------------------------

    output_dir = Path(
        "data"
    ) / f"{dataset_label}Embeddings"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_path = (
        output_dir
        / f"{dataset_name}_train_embeddings.npy"
    )

    test_path = (
        output_dir
        / f"{dataset_name}_test_embeddings.npy"
    )

    np.save(
        train_path,
        train_output
    )

    np.save(
        test_path,
        test_output
    )

    # --------------------------------------------------
    # 8. Summary
    # --------------------------------------------------

    print("\n================================")
    print("Embedding generation finished")
    print("================================")

    print(
        f"Train embeddings: "
        f"{train_embeddings.shape}"
    )

    print(
        f"Test embeddings: "
        f"{test_embeddings.shape}"
    )

    print(
        f"Train saved to: {train_path}"
    )

    print(
        f"Test saved to: {test_path}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])