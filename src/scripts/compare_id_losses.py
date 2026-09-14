"""Train CE, CE+MLE, CE+TwoNN and CE+smooth-CorrInt with paired seeds."""
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_geometric
import yaml

from compare_losses import (ROOT, DATASETS, seed_all, read_embeddings, build_graph,
                            training_graph, batches, write_rows)
from models.classifier import GNNModel
from training.id_regularizers import id_penalty

ARMS = ("ce", "mle", "twonn", "corrint")


@torch.no_grad()
def evaluate(model, graph, nodes, args, layers, device):
    """Prediction metrics only: no dimension estimators in evaluation."""
    model.eval()
    ce_sum, correct, count = 0., 0, 0
    for batch in batches(graph, nodes, args, layers):
        batch = batch.to(device)
        logits = model(batch)
        labels = batch.y[:batch.batch_size]
        ce_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    return {"ce": ce_sum / count, "accuracy": correct / count}


def objective(logits, features, labels, arm, args):
    ce = F.cross_entropy(logits, labels)
    # No ID computation in CE or lambda=0 controls. All arms use the same
    # forward call and consume the same RNG stream regardless of loss type.
    penalty = features.sum() * 0
    if arm != "ce" and args.lambda_id != 0:
        penalty = id_penalty(features, arm, mle_k=args.mle_k,
                             twonn_discard=args.twonn_discard,
                             corr_k1=args.corr_k1, corr_k2=args.corr_k2,
                             corr_temperature=args.corr_temperature)
    return ce + args.lambda_id * penalty, penalty


def run_dataset(name, args, model_args, training):
    out = args.output / name
    out.mkdir(parents=True, exist_ok=True)
    arrays, sources = read_embeddings(args.data_root, name, args.limit)
    manifest = {
        "version": 2, "id_direction": "lower", "objective": "ce_plus_lambda_log_id",
        "dataset": name, "arms": list(ARMS), "sources": sources,
        "model": model_args, "training": training,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                      if k not in ("resume", "datasets", "output")},
        "torch": str(torch.__version__), "pyg": torch_geometric.__version__,
        "protocol": "induced train / transductive test; final epoch; no ID evaluation"
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"{out} exists; use --resume or a fresh --output")
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"Changed data/settings in {out}; choose a fresh --output")
    elif any(out.iterdir()):
        raise FileExistsError(f"{out} contains files without a manifest; choose a fresh --output")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    cache = out / "graph.pt"
    if cache.exists():
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
        for arm in ARMS:
            run = out / f"{arm}_seed{seed}"
            run.mkdir(exist_ok=True)
            if args.resume and (run / "complete.json").exists():
                print(f"Skipping completed {name}/{arm}/seed{seed}", flush=True)
                continue
            seed_all(seed)
            model = GNNModel(**dims).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"], weight_decay=0.)
            history = []
            for epoch in range(1, training["num_epochs"] + 1):
                seed_all(seed * 100000 + epoch)
                model.train()
                total, penalty_total, count = 0., 0., 0
                for batch in batches(train_graph, train_nodes, args, layers, train=True):
                    batch = batch.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits, features = model(batch, return_embeddings=True)
                    loss, penalty = objective(logits, features, batch.y[:batch.batch_size], arm, args)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss: {name}/{arm}/{seed}/{epoch}")
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(features)
                    penalty_total += penalty.item() * len(features)
                    count += len(features)
                seed_all(seed * 100000 + 90000)
                train = evaluate(model, train_graph, train_nodes, args, layers, device)
                seed_all(seed * 100000 + 90001)
                test = evaluate(model, graph, test_nodes, args, layers, device)
                row = {"dataset": name, "method": arm, "seed": seed, "epoch": epoch,
                       "id_direction": "lower", "lambda_id": args.lambda_id if arm != "ce" else 0.,
                       "train_objective": total / count, "train_penalty": penalty_total / count,
                       **{f"train_{k}": v for k, v in train.items()},
                       **{f"test_{k}": v for k, v in test.items()}}
                history.append(row)
                write_rows(run / "history.csv", history)
                print(f"{name} {arm} seed={seed} epoch={epoch}: "
                      f"test CE={test['ce']:.4f}, accuracy={test['accuracy']:.4f}", flush=True)
            temporary = run / "final.tmp"
            torch.save({"state_dict": model.cpu().state_dict(), "model_args": dims,
                        "seed": seed, "method": arm, "epoch": epoch,
                        "experiment": manifest}, temporary)
            temporary.replace(run / "final.pt")
            (run / "complete.json").write_text(json.dumps(history[-1], indent=2))
        # A plot becomes available after each complete four-arm seed group.
        from plot_id_loss_comparison import plot_results
        plot_results(args.output)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--output", type=Path, default=ROOT / "results/id_loss_comparison_lower_id")
    p.add_argument("--model-config", type=Path, default=ROOT / "config/models/gnn.yaml")
    p.add_argument("--training-config", type=Path, default=ROOT / "config/gnn_training_config.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--lambda-id", type=float, default=0.01)
    p.add_argument("--mle-k", type=int, default=20)
    p.add_argument("--twonn-discard", type=float, default=0.1)
    p.add_argument("--corr-k1", type=int, default=10)
    p.add_argument("--corr-k2", type=int, default=20)
    p.add_argument("--corr-temperature", type=float, default=0.1)
    p.add_argument("--knn", type=int, default=100)
    p.add_argument("--trees", type=int, default=50)
    p.add_argument("--graph-seed", type=int, default=0)
    p.add_argument("--graph-backend", choices=["annoy", "exact"], default="annoy")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cpu-threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--train-neighbors", type=int, default=10)
    p.add_argument("--eval-neighbors", type=int, default=30)
    p.add_argument("--limit", type=int, default=0, help="Random examples per split for smoke tests; 0 uses all")
    p.add_argument("--full-batch", action="store_true", help="Small CPU tests without PyG sampling extensions")
    p.add_argument("--resume", action="store_true", help="Skip complete arms; restart interrupted arms")
    args = p.parse_args()
    model = yaml.safe_load(args.model_config.read_text())
    training = yaml.safe_load(args.training_config.read_text())
    if model["model"] != "GNNModel" or training["loss"] != "torch_ce_loss":
        p.error("Requires repository GNNModel and torch_ce_loss configs")
    if args.epochs is not None:
        training["num_epochs"] = args.epochs
    args.batch_size = training["batch_size"] if args.batch_size is None else args.batch_size
    training["batch_size"] = args.batch_size
    if min(args.knn, args.trees, args.cpu_threads, training["num_epochs"]) < 1:
        p.error("Counts and epochs must be positive")
    if args.mle_k < 2 or not 1 <= args.corr_k1 < args.corr_k2:
        p.error("Require mle-k >= 2 and 1 <= corr-k1 < corr-k2")
    if args.batch_size <= max(args.mle_k, args.corr_k2):
        p.error("batch-size must exceed mle-k and corr-k2; partial final batches use reduced k")
    if not math.isfinite(args.lambda_id) or args.lambda_id < 0:
        p.error("lambda-id must be finite and nonnegative")
    if not 0 <= args.twonn_discard < 1 or not math.isfinite(args.corr_temperature) or args.corr_temperature <= 0:
        p.error("Require 0 <= twonn-discard < 1 and finite positive corr-temperature")
    if args.limit < 0 or (args.limit and args.limit <= max(args.mle_k, args.corr_k2)):
        p.error("limit must be 0 or exceed mle-k and corr-k2")
    if args.full_batch and (not args.limit or args.limit > 2000):
        p.error("full-batch requires a positive limit <= 2000")
    if args.train_neighbors < 1 or (args.eval_neighbors != -1 and args.eval_neighbors < 1):
        p.error("train-neighbors must be positive; eval-neighbors positive or -1")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or max(args.seeds) >= 40000:
        p.error("Seeds must be unique integers in [0, 40000)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA is unavailable; use matching CUDA PyTorch or --device cpu")
    if not args.full_batch:
        from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE
        if not (WITH_PYG_LIB or WITH_TORCH_SPARSE):
            p.error("Install matching pyg-lib or torch-sparse for neighbor sampling")
    return args, model["args"], training


if __name__ == "__main__":
    args, model_args, training = parse_args()
    torch.set_num_threads(args.cpu_threads)
    for name in args.datasets:
        run_dataset(name, args, model_args, training)
