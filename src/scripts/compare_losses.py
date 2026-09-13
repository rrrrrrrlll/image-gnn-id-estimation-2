"""Paired CE versus CE + LDReg experiments from existing SMSL embeddings."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected

from models.classifier import GNNModel
from training.id_loss import local_log_id

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {"mnist": "MNIST", "fmnist": "FMNIST", "pathmnist": "PathMNIST"}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_embeddings(root, name, limit=0):
    arrays, sources = [], []
    for split in ("train", "test"):
        path = root / f"{DATASETS[name]}Embeddings/smsl_embeddings/{name}_smsl_{split}_embeddings.npy"
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        if a.ndim != 2 or a.shape[1] < 3 or len(a) < 3:
            raise ValueError(f"Invalid SMSL shape: {path}: {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError(f"Nonfinite data in {path}")
        if not np.isin(a[:, -1], [0, 1]).all() or not np.all(a[:, -2] == a[:, -2].astype(np.int64)):
            raise ValueError(f"Expected [features | integer class | binary SMSL flag]: {path}")
        # Only active rows: do not accidentally include the opposite split from
        # a full SMSL view. The supplied six files have all flags equal to one.
        a = np.asarray(a[a[:, -1] == 1])
        if limit:
            idx = np.random.default_rng(0).permutation(len(a))[:limit]
            a = a[idx]
        if len(a) < 3:
            raise ValueError(f"At least three active rows required in {path}")
        arrays.append(a)
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        sources.append({"path": str(path.resolve()), "sha256": digest})
    if arrays[0].shape[1] != arrays[1].shape[1]:
        raise ValueError("Train and test feature dimensions differ")
    classes = np.unique(arrays[0][:, -2])
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Train labels must cover contiguous classes starting at zero; increase smoke limit")
    if not np.isin(arrays[1][:, -2], classes).all():
        raise ValueError("Test contains a class missing from train")
    return arrays, sources


def build_graph(arrays, args):
    a = np.concatenate(arrays)
    x = np.ascontiguousarray(a[:, :-2], dtype=np.float32)
    n, k = len(x), min(args.knn, len(x) - 1)
    print(f"Building {args.graph_backend} graph: {n} nodes, {x.shape[1]} features, k={k}", flush=True)
    neighbors = np.empty((n, k), dtype=np.int64)
    distances = np.empty((n, k), dtype=np.float32)
    if args.graph_backend == "annoy":
        from annoy import AnnoyIndex
        index = AnnoyIndex(x.shape[1], "euclidean")
        index.set_seed(args.graph_seed)
        for i, row in enumerate(x):
            index.add_item(i, row)
        index.build(args.trees, n_jobs=1)
        for i in range(n):
            ids, ds = index.get_nns_by_item(i, k + 1, include_distances=True)
            pairs = [(j, d) for j, d in zip(ids, ds) if j != i][:k]
            if len(pairs) != k:
                raise RuntimeError("Annoy returned too few neighbors")
            neighbors[i], distances[i] = zip(*pairs)
    else:
        from sklearn.neighbors import NearestNeighbors
        index = NearestNeighbors(n_neighbors=k + 1, n_jobs=args.cpu_threads).fit(x)
        for start in range(0, n, 512):
            ds, ids = index.kneighbors(x[start:start + 512])
            for offset in range(len(ids)):
                keep = ids[offset] != start + offset
                neighbors[start + offset] = ids[offset][keep][:k]
                distances[start + offset] = ds[offset][keep][:k]
    edge_index = torch.tensor(np.stack((np.repeat(np.arange(n), k), neighbors.ravel())))
    weights = torch.exp(-torch.from_numpy(distances.ravel()) / 25.0)
    threshold = float(np.percentile(weights.numpy(), 50))
    weights[weights < threshold] = 0
    edge_index, weights = to_undirected(edge_index, weights, num_nodes=n, reduce="mean")
    keep = weights > 0
    graph = Data(x=torch.from_numpy(x), y=torch.tensor(a[:, -2], dtype=torch.long),
                 edge_index=edge_index[:, keep], edge_weight=weights[keep])
    graph.n_train = len(arrays[0])
    graph.threshold = threshold
    return graph


def training_graph(graph):
    # Match the existing gap-curve trainer's induced training graph. Test labels
    # and test nodes are absent from all training forward passes (including BN).
    n = graph.n_train
    keep = (graph.edge_index[0] < n) & (graph.edge_index[1] < n)
    return Data(x=graph.x[:n], y=graph.y[:n], edge_index=graph.edge_index[:, keep],
                edge_weight=graph.edge_weight[keep])


def batches(graph, nodes, args, layers, train=False):
    if args.full_batch:
        # Smoke/testing path without compiled PyG neighbor-sampling extensions.
        # Relabel so the scored nodes occupy the initial batch_size positions.
        selected = torch.zeros(graph.num_nodes, dtype=torch.bool)
        selected[nodes] = True
        order = torch.cat((nodes, torch.where(~selected)[0]))
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order))
        yield Data(x=graph.x[order], y=graph.y[order],
                   edge_index=inverse[graph.edge_index], edge_weight=graph.edge_weight,
                   batch_size=len(nodes))
        return
    loader = NeighborLoader(graph, input_nodes=nodes,
                            num_neighbors=[args.train_neighbors if train else args.eval_neighbors] * layers,
                            batch_size=args.batch_size, shuffle=train, num_workers=0)
    yield from loader


@torch.no_grad()
def evaluate(model, graph, nodes, args, layers, device):
    model.eval()
    ce_sum, correct, count, id_sum, id_count = 0., 0, 0, 0., 0
    for batch in batches(graph, nodes, args, layers):
        batch = batch.to(device)
        logits, z = model(batch, return_embeddings=True)
        labels = batch.y[:batch.batch_size]
        ce_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
        if len(z) >= 3:
            id_sum += local_log_id(z, args.id_k).sum().item()
            id_count += len(z)
    return {"ce": ce_sum / count, "accuracy": correct / count,
            "log_id": id_sum / id_count if id_count else float("nan")}


def write_rows(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(name, args, model_args, training):
    out = args.output / name
    out.mkdir(parents=True, exist_ok=True)
    arrays, sources = read_embeddings(args.data_root, name, args.limit)
    graph_settings = {k: getattr(args, k) for k in ("knn", "graph_seed", "trees", "graph_backend", "limit")}
    graph_key = {"sources": sources, "settings": graph_settings, "version": 1}
    manifest = {"dataset": name, "graph": graph_key, "model": model_args, "training": training,
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                              if k not in ("resume", "datasets", "output")},
                "torch": torch.__version__, "protocol": "induced-train/transductive-test; final epoch"}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"{out} exists; use a fresh --output or --resume")
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"Configuration/data changed for {out}; choose a new --output")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    cache = out / "graph.pt"
    if cache.exists():
        # This file is generated locally by this runner; never load third-party caches.
        graph = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        graph = build_graph(arrays, args)
        temporary = cache.with_suffix(".tmp")
        torch.save(graph, temporary)
        temporary.replace(cache)
    del arrays
    train_graph = training_graph(graph)
    train_nodes = torch.arange(graph.n_train)
    test_nodes = torch.arange(graph.n_train, graph.num_nodes)
    dims = dict(model_args, in_features=graph.x.shape[1], out_size=int(train_graph.y.max()) + 1)
    layers = dims["layers"]
    device = torch.device(args.device)
    for seed in args.seeds:
        for method, coefficient in (("ce", 0.), ("ldreg", args.lambda_id)):
            run = out / f"{method}_seed{seed}"
            run.mkdir(exist_ok=True)
            if args.resume and (run / "complete.json").exists():
                continue
            seed_all(seed)
            model = GNNModel(**dims).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"], weight_decay=0.)
            history = []
            for epoch in range(1, training["num_epochs"] + 1):
                # Reset the same stream per epoch for paired sampling and dropout.
                seed_all(seed * 100000 + epoch)
                model.train()
                total, count = 0., 0
                for batch in batches(train_graph, train_nodes, args, layers, train=True):
                    batch = batch.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits, z = model(batch, return_embeddings=True)
                    ce = F.cross_entropy(logits, batch.y[:batch.batch_size])
                    penalty = -local_log_id(z, args.id_k).mean() if len(z) >= 3 else z.sum() * 0
                    loss = ce + coefficient * penalty
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss: {name}/{method}/{seed}/{epoch}")
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(z)
                    count += len(z)
                seed_all(seed * 100000 + 90000)
                train = evaluate(model, train_graph, train_nodes, args, layers, device)
                seed_all(seed * 100000 + 90001)
                test = evaluate(model, graph, test_nodes, args, layers, device)
                row = {"dataset": name, "method": method, "seed": seed, "epoch": epoch,
                       "lambda_id": coefficient, "train_objective": total / count,
                       **{f"train_{k}": v for k, v in train.items()},
                       **{f"test_{k}": v for k, v in test.items()}}
                history.append(row)
                write_rows(run / "history.csv", history)
                print(f"{name} {method} seed={seed} epoch={epoch}: test CE={test['ce']:.4f}, accuracy={test['accuracy']:.4f}", flush=True)
            torch.save({"state_dict": model.cpu().state_dict(), "model_args": dims,
                        "seed": seed, "method": method, "epoch": epoch}, run / "final.pt")
            (run / "complete.json").write_text(json.dumps(history[-1], indent=2))
    from plot_loss_comparison import plot_results
    plot_results(args.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--output", type=Path, default=ROOT / "results/loss_comparison")
    p.add_argument("--model-config", type=Path, default=ROOT / "config/models/gnn.yaml")
    p.add_argument("--training-config", type=Path, default=ROOT / "config/gnn_training_config.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--lambda-id", type=float, default=0.01)
    p.add_argument("--id-k", type=int, default=20)
    p.add_argument("--knn", type=int, default=100)
    p.add_argument("--trees", type=int, default=50)
    p.add_argument("--graph-seed", type=int, default=0)
    p.add_argument("--graph-backend", choices=["annoy", "exact"], default="annoy")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cpu-threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--train-neighbors", type=int, default=10)
    p.add_argument("--eval-neighbors", type=int, default=30, help="Per-hop cap; -1 for full neighborhoods")
    p.add_argument("--limit", type=int, default=0, help="Smoke test: random rows per split (0 = all)")
    p.add_argument("--full-batch", action="store_true", help="Small CPU tests without PyG sampling extensions")
    p.add_argument("--resume", action="store_true", help="Skip complete runs; restart incomplete runs deterministically")
    args = p.parse_args()
    model = yaml.safe_load(args.model_config.read_text())
    training = yaml.safe_load(args.training_config.read_text())
    if model["model"] != "GNNModel" or training["loss"] != "torch_ce_loss":
        p.error("Comparison requires repository GNNModel and torch_ce_loss configs")
    if args.epochs is not None:
        training["num_epochs"] = args.epochs
    args.batch_size = args.batch_size if args.batch_size is not None else training["batch_size"]
    training["batch_size"] = args.batch_size
    if min(args.batch_size, args.id_k, args.knn, args.trees, args.cpu_threads, training["num_epochs"]) < 1 or args.batch_size < 3 or args.id_k < 2:
        p.error("Positive counts required; batch-size >= 3, id-k >= 2")
    if args.lambda_id < 0 or not np.isfinite(args.lambda_id) or args.limit < 0:
        p.error("lambda-id must be finite/nonnegative; limit must be nonnegative")
    if args.train_neighbors < 1 or (args.eval_neighbors != -1 and args.eval_neighbors < 1):
        p.error("train-neighbors must be positive; eval-neighbors must be positive or -1")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or max(args.seeds) >= 40000:
        p.error("Seeds must be unique integers in [0, 40000)")
    if args.full_batch and (not args.limit or args.limit > 2000):
        p.error("--full-batch requires --limit between 3 and 2000 to bound memory")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA is unavailable in this environment; install matching CUDA PyTorch or use --device cpu")
    if not args.full_batch:
        from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE
        if not (WITH_PYG_LIB or WITH_TORCH_SPARSE):
            p.error("Neighbor sampling requires pyg-lib or torch-sparse matching PyTorch/CUDA. See README; use --full-batch --limit 128 for a smoke test.")
    torch.set_num_threads(args.cpu_threads)
    for name in args.datasets:
        run_dataset(name, args, model["args"], training)


if __name__ == "__main__":
    main()
