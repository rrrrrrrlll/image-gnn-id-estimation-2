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
        # n_trees=100000 previously sat here. Under the old (buggy) build order
        # in kNNModel.__init__ (build() called before add_item()), the tree
        # count never actually mattered -- Annoy built empty trees regardless
        # of n_trees, so an absurdly high value was silently free. Now that
        # add_item() runs first (see the fix in models/models.py), Annoy
        # genuinely builds this many trees over the full dataset, and 100000
        # is far outside Annoy's normal range (typically tens to a few
        # hundred) -- it OOM-killed every dataset's SLURM job once the build
        # order was corrected. 50 is a standard, well-supported value.
        n_trees=50
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

    # Use this dataset's own 50th-percentile (median) edge weight as the
    # threshold instead of a fixed constant or a mean-based statistic. A
    # fixed cutoff (previously 0.75) was tuned around datasets whose typical
    # nearest-neighbor distance is small (MNIST, FMNIST, FER2013, PathMNIST);
    # CIFAR10 and CelebA have a much larger typical distance, so a fixed 0.75
    # zeroed out the large majority of CIFAR10's edges. Mean-based thresholds
    # (plain mean, then mean - std) were dataset-adaptive but still sensitive
    # to the shape of each dataset's distribution -- for MNIST/FMNIST/PathMNIST
    # the mean landed at or near the distribution's own peak, and both
    # mean-based versions were additionally skewed upward by a spurious
    # weight=1.0 self-loop edge that used to be included for every node (see
    # the fix in kNNModel.__call__, models/models.py). A percentile threshold
    # sidesteps both problems: it guarantees a specific, dataset-independent
    # fraction of edges survive (here, the top half by weight) regardless of
    # the distribution's shape.
    threshold = float(np.percentile(edge_weight.numpy(), 50))
    print(f"Using 50th percentile (median) edge weight as threshold: {threshold:.4f}")
    edge_weight[edge_weight < threshold] = 0.0

    edge_index, edge_weight = to_undirected(
        edge_index,
        edge_weight,
        reduce='mean'
    )

    # to_undirected() only symmetrizes edges and merges duplicates by weight -- it does
    # NOT drop the edges the threshold above just zeroed out. Without this filter, the
    # graph saved to disk keeps its full, untrimmed k-neighbor topology forever: roughly
    # half of every node's structural neighbors in edge_index are "dead" (zero weight)
    # even after thresholding. NeighborLoader (used for both sample_subgraph()'s bounded
    # num_neighbors=[10, 10] training sampling and the unbounded [-1, -1] test-time
    # evaluation) samples purely from edge_index topology and has no notion of
    # edge_weight, so training's bounded 10-per-hop draw was wasting roughly half its
    # budget on edges that contribute nothing to GCNConv's weighted aggregation --
    # confirmed as the likely cause of GNN accuracy trailing the graph-free MLP baseline
    # by a wide margin, and of the generalization gap growing with graph size instead of
    # shrinking. Actually removing the zero-weight edges here makes the saved graph's
    # structure match what the threshold was meant to produce.
    keep = edge_weight > 0
    edge_index = edge_index[:, keep]
    edge_weight = edge_weight[keep]

    plt.figure(figsize=(7, 4))
    plt.hist(edge_weight.numpy(), bins=100)
    plt.title(f"Edge Weights After Thresholding (< {threshold:.4f} = median set to 0, edges dropped)")
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
    print(f"Edge weight threshold (50th percentile / median): {threshold:.4f}")
    print(f"Edges: {edge_index.shape[1]}")
    print(f"Train graph: {train_output}")
    print(f"Test graph:  {test_output}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
