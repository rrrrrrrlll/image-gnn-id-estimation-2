import sys
import yaml
import argparse
import numpy as np

from pathlib import Path


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset_config",
        type=str,
        required=True,
        help="Dataset configuration YAML file"
    )

    return parser.parse_args(args)


def main(sys_args):
    args = parse_args(sys_args)

    # --------------------------------------------------
    # 1. Read dataset name from dataset config
    # --------------------------------------------------

    with open(args.dataset_config, "r") as f:
        dataset_config = yaml.safe_load(f)

    dataset_class = dataset_config["train"]["dataset_class"]

    # Examples:
    # MNISTDataset -> MNIST
    # FMNISTDataset -> FMNIST
    # PathMNISTDataset -> PathMNIST
    dataset_label = dataset_class.replace("Dataset", "")
    dataset_name = dataset_label.lower()

    print(f"Dataset: {dataset_label}")

    # --------------------------------------------------
    # 2. Locate VAE embeddings
    # --------------------------------------------------

    embedding_dir = (
        Path("data")
        / f"{dataset_label}Embeddings"
    )

    train_path = (
        embedding_dir
        / f"{dataset_name}_train_embeddings.npy"
    )

    test_path = (
        embedding_dir
        / f"{dataset_name}_test_embeddings.npy"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train embeddings not found: {train_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test embeddings not found: {test_path}"
        )

    print(f"Loading train embeddings: {train_path}")
    print(f"Loading test embeddings:  {test_path}")

    train_data = np.load(train_path)
    test_data = np.load(test_path)

    print(f"Train input shape: {train_data.shape}")
    print(f"Test input shape:  {test_data.shape}")

    # --------------------------------------------------
    # 3. Add SMSL membership column
    #
    # Input:
    # [embedding..., class_label]
    #
    # Output:
    # [embedding..., class_label, membership]
    #
    # membership = 1 -> visible / active set
    # membership = 0 -> hidden / opposite set
    # --------------------------------------------------

    train_ones = np.ones(
        (train_data.shape[0], 1),
        dtype=np.int32
    )

    train_zeros = np.zeros(
        (train_data.shape[0], 1),
        dtype=np.int32
    )

    test_ones = np.ones(
        (test_data.shape[0], 1),
        dtype=np.int32
    )

    test_zeros = np.zeros(
        (test_data.shape[0], 1),
        dtype=np.int32
    )

    # Training view
    train_tagged = np.concatenate(
        [train_data, train_ones],
        axis=1
    )

    test_hidden_for_train = np.concatenate(
        [test_data, test_zeros],
        axis=1
    )

    train_full = np.concatenate(
        [
            train_tagged,
            test_hidden_for_train
        ],
        axis=0
    )

    # Testing view
    test_tagged = np.concatenate(
        [test_data, test_ones],
        axis=1
    )

    train_hidden_for_test = np.concatenate(
        [train_data, train_zeros],
        axis=1
    )

    test_full = np.concatenate(
        [
            test_tagged,
            train_hidden_for_test
        ],
        axis=0
    )

    # --------------------------------------------------
    # 4. Output directory
    # --------------------------------------------------

    output_dir = (
        embedding_dir
        / "smsl_embeddings"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Keep "_pca_" in the filename for compatibility
    # with the existing repository configuration.
    train_output_path = (
        output_dir
        / f"{dataset_name}_smsl_train_pca_embeddings.npy"
    )

    test_output_path = (
        output_dir
        / f"{dataset_name}_smsl_test_pca_embeddings.npy"
    )

    train_full_output_path = (
        output_dir
        / f"{dataset_name}_smsl_train_full_pca_embeddings.npy"
    )

    test_full_output_path = (
        output_dir
        / f"{dataset_name}_smsl_test_full_pca_embeddings.npy"
    )

    # --------------------------------------------------
    # 5. Save
    # --------------------------------------------------

    np.save(
        train_output_path,
        train_tagged
    )

    np.save(
        test_output_path,
        test_tagged
    )

    np.save(
        train_full_output_path,
        train_full
    )

    np.save(
        test_full_output_path,
        test_full
    )

    # --------------------------------------------------
    # 6. Summary
    # --------------------------------------------------

    print("\n================================")
    print("SMSL parsing finished")
    print("================================")

    print(
        f"Train parsed shape: {train_tagged.shape}"
    )

    print(
        f"Test parsed shape: {test_tagged.shape}"
    )

    print(
        f"Train full shape: {train_full.shape}"
    )

    print(
        f"Test full shape: {test_full.shape}"
    )

    print("\nSaved:")
    print(train_output_path)
    print(test_output_path)
    print(train_full_output_path)
    print(test_full_output_path)


if __name__ == "__main__":
    main(sys.argv[1:])