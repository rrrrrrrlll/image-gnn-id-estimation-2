import sys
import yaml
import pickle
import argparse

import torch

import numpy as np

import matplotlib.pyplot as plt

from models.models import kNNModel

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected

from dataclasses import dataclass

from tqdm import tqdm
from time import time
from pathlib import Path


def smart_load(path):
    path = Path(path)
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif path.suffix == ".npy":
        return np.load(path, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


@dataclass
class Embedding:
    dataset: np.array
    train: bool


def build_datasets(ds_config):
    datasets = []
    for ds_type, configs in ds_config.items():
        datasets.append(
            Embedding(
                dataset=smart_load(configs["base_dir"]),
                train=True if ds_type == "train" else False
            )
        )
    return datasets


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
        "-k",
        "--knn",
        type=str,
        help="Number of sampled neighbors",
        required=True
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name, e.g. mnist, fmnist, pathmnist"
    )

    return parser.parse_args(args)


def build_edge_index(node_idx, neighb_idxs):
    source = torch.full((len(neighb_idxs),), node_idx, dtype=torch.long)
    target = torch.tensor(neighb_idxs, dtype=torch.long)
    return torch.stack([source, target], dim=0)


def main(sys_args):
    
    args = parse_args(sys_args)
    dataset_name = args.dataset_name.lower()
    dataset_label_map = {
        "mnist": "MNIST",
        "fmnist": "FMNIST",
        "octmnist": "OCTMNIST",
        "pathmnist": "PathMNIST",
        "cifar10": "CIFAR10",
        "fer2013": "FER2013",
        "celeba_gb": "CELEBA",
        "celeba_sb": "CELEBA",
    }

    if dataset_name not in dataset_label_map:
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    dataset_label = dataset_label_map[dataset_name]
    k = int(args.knn)
    ker_width = 5

    with open(args.dataset_config, 'r') as f:
        ds_config = yaml.safe_load(f)

    ds = build_datasets(ds_config)
    # ds[0].dataset = ds[0].dataset[:2500, :]
    # ds[1].dataset = ds[1].dataset[:2500, :]
    full_ds = np.append(ds[0].dataset, ds[1].dataset, 0)
    model = kNNModel(
        ds=full_ds[:, :-2],
        #n_trees=50
        n_trees=100000
    )

    x = torch.tensor(full_ds[:, :-2], dtype=torch.float)
    y = torch.tensor(full_ds[:, -2], dtype=torch.long)

    edge_indices = []
    edge_weights = []
    for i in tqdm(range(len(model.indexes))):
        neighbors, distances = model(i, k)
        edge_indices.append(build_edge_index(i, neighbors))
        edge_weights.append(torch.exp(-torch.tensor(distances) / (ker_width ** 2)))

    edge_index = torch.cat(edge_indices, dim=1)
    edge_weight = torch.cat(edge_weights, dim=0)

    raw_weights = torch.cat(edge_weights, dim=0).numpy()

    plt.figure(figsize=(7, 4))
    plt.hist(raw_weights, bins=100)
    plt.title("Edge Weights (Gaussian Kernel) Before Thresholding")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
    f"{dataset_name}_edge_weights_before_thresholding.png"
    )
    plt.show()

    edge_weight = edge_weight.view(-1)
    edge_weight[edge_weight < 0.75] = 0.0

    edge_index, edge_weight = to_undirected(
        edge_index,
        edge_weight,
        reduce='mean'
    )

    plt.figure(figsize=(7, 4))
    plt.hist(edge_weight.numpy(), bins=100)
    plt.title("Edge Weights After Thresholding (< 0.75 set to 0)")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
    f"{dataset_name}_edge_weights_after_thresholding.png"
    )
    plt.show()

    train_mask = torch.zeros(full_ds.shape[0], dtype=torch.bool)
    train_mask[:ds[0].dataset.shape[0]] = True
    test_mask = ~train_mask

    train_graph_data = Data(x, edge_index, edge_weight=edge_weight, y=y, mask=train_mask)
    test_graph_data = Data(x, edge_index, edge_weight=edge_weight, y=y, mask=test_mask)

    output_dir = Path(
        "data"
    ) / f"{dataset_label}Graph"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_output = (
        output_dir
        / f"{dataset_name}_train_knn_graph-{k}.pkl"
    )

    test_output = (
        output_dir
        / f"{dataset_name}_test_knn_graph-{k}.pkl"
    )

    torch.save(
        train_graph_data,
        train_output
    )

    torch.save(
        test_graph_data,
        test_output
    )

    print("\n==============================")
    print("KNN graph generation finished")
    print("==============================")
    print(f"Nodes: {full_ds.shape[0]}")
    print(f"Features per node: {x.shape[1]}")
    print(f"Edges: {edge_index.shape[1]}")
    print(f"Train graph: {train_output}")
    print(f"Test graph:  {test_output}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
