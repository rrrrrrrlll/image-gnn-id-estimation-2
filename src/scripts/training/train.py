import sys
import csv
import yaml
import wandb

import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm
from datetime import datetime

from pathlib import Path

from torch.utils.data import Subset

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborLoader

from utils.metrics import torch_rmse, torch_vae_loss, torch_vqvae_loss, torch_ce_loss
from data_preproc.datasets import build_datasets
from models.models import (
    VAEModel, 
    CNNVAEModel, 
    AEModel, 
    VQVAEModel, 
    GNNModel, 
    GNNProjModel,
    GNN,
    DGMGNNModel, 
    MLPModel
)


class Trainer():

    def __init__(
        self, 
        dataset_config, 
        training_config, 
        model_config,
        save_model=False
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_ds, self.test_ds = build_datasets(dataset_config)
        self.training_config = training_config
        self.model_config = model_config

        self.dataset_name = (
            dataset_config["train"]["dataset_class"]
            .replace("Dataset", "")
            .lower()
        )

    def run_batch(self, model, dl, mode):        
        pass

    def run_epoch(self, epochs, model, dl, mode):
        pass

    def model_checkpoint(self, model, test_loss, epoch):
        checkpoint_dir = (
            Path("src/scripts/checkpoints") / self.dataset_name
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        run_id = (
            wandb.run.id
            if wandb.run is not None
            else "local"
        )

        sweep_id = (
            wandb.run.sweep_id
            if wandb.run is not None
            else None
        )

        checkpoint_path = (
            checkpoint_dir
            / f"{self.dataset_name}_{run_id}_best.pt"
        )

        checkpoint = {
            "state_dict": model.state_dict(),
            "model_config": self.model_config,
            "training_config": self.training_config,
            "dataset_name": self.dataset_name,
            "epoch": epoch,
            "test_loss": float(test_loss),
            "wandb_run_id": run_id,
            "sweep_id": sweep_id,
        }

        torch.save(
            checkpoint,
            checkpoint_path
        )

        print(
            f"Saved best checkpoint: {checkpoint_path}"
        )

        return checkpoint_path

    def sample_subgraph(self, graph_ds, n_target, batch_size, seed=None):
        """
        Samples a fixed-size induced subgraph from graph_ds: n_target nodes
        drawn (without replacement) from the original training nodes, plus
        every edge of the base graph whose endpoints are both included.

        This is the same node/edge sampling parse_grow_graph() always did;
        it is factored out here so train_eval_gap_curve() can call it once
        per (graph size, seed) instead of once per growing-sequence step.
        Pass seed for a reproducible sample across independent runs.
        """
        generator = None

        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        # Only sample from original training nodes
        eligible_idx = torch.where(
            graph_ds.data.mask.bool()
        )[0]

        n_target = min(n_target, eligible_idx.shape[0])

        if generator is not None:
            perm = torch.randperm(
                eligible_idx.shape[0],
                generator=generator
            )[:n_target]
        else:
            perm = torch.randperm(
                eligible_idx.shape[0]
            )[:n_target]

        idx = eligible_idx[perm]

        mask = torch.zeros_like(
            graph_ds.data.mask,
            dtype=torch.bool
        )

        mask[idx] = True

        edge_mask = (
            mask[graph_ds.data.edge_index[0]]
            & mask[graph_ds.data.edge_index[1]]
        )

        sub_data = Data(
            x=graph_ds.data.x,
            y=graph_ds.data.y,
            edge_index=graph_ds.data.edge_index[:, edge_mask],
            edge_weight=graph_ds.data.edge_weight[edge_mask],
            mask=mask,
            m=mask.sum().item()
        )

        loader = NeighborLoader(
            sub_data,
            num_neighbors=[10, 10],
            batch_size=batch_size,
            input_nodes=sub_data.mask.bool()
        )

        return loader

    def parse_grow_graph(self, graph_ds, batch_size, size_fractions=(0.1, 0.25, 0.5, 0.75, 1.0)):
        """
        Growing-graph schedule for train_eval_grow_graph() (Step 11 /
        the base paper's Corollary IV.7 procedure). Was hardcoded to
        n0=6000, n_increases=1, increase_rate=5000 -- a FIXED absolute
        node count that (a) never actually grew, since n_increases=1
        means this loop only ever ran once, and (b) would mean wildly
        different relative sizes across this fork's datasets today
        (6000 nodes is 10% of MNIST's ~60k eligible nodes but under 4%
        of CelebA's ~163k). Switched to fractions of the eligible node
        count, matching train_eval_gap_curve()'s own size_fractions
        convention (same defaults) so the two procedures are run at
        directly comparable sizes across every dataset.
        """
        eligible_idx = torch.where(
            graph_ds.data.mask.bool()
        )[0]
        n_eligible = eligible_idx.shape[0]

        grow_graph_loader = []

        for fraction in size_fractions:
            m = max(1, round(fraction * n_eligible))

            loader = self.sample_subgraph(
                graph_ds,
                m,
                batch_size
            )

            grow_graph_loader.append(loader)

        return grow_graph_loader

    def train(self):
        train_dl = DataLoader(
            dataset=self.train_ds, 
            batch_size=self.training_config["batch_size"], 
            shuffle=True, 
            num_workers=4
        )

        test_dl = DataLoader(
            dataset=self.test_ds,
            batch_size=self.training_config["batch_size"], 
            shuffle=True, 
            num_workers=4
        )

        lr = self.training_config["learning_rate"]
        epochs = self.training_config["num_epochs"] 

        model : nn.Module = getattr(sys.modules[__name__], self.model_config["model"])
        model = model(**model.pre_init(self.model_config["args"])).to(self.device)

        loss = getattr(sys.modules[__name__], self.training_config["loss"])
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        best_test_loss = float("inf")
        for epoch in tqdm(range(epochs)):
            test_losses = []
            train_losses = []

            model.train()
            for i, train_batch in tqdm(enumerate(train_dl)):
                batch = train_batch.to(self.device)

                # Forward pass
                y_hat = model(batch)

                # Compute loss
                J = loss(batch.y, y_hat)
                train_losses.append(J.detach().cpu().numpy())

                # Backward pass
                J.backward()

                # Optimization step
                optimizer.step()

                optimizer.zero_grad()

            model.eval()
            with torch.no_grad():
                for i, test_batch in enumerate(test_dl):
                    batch = test_batch.to(self.device)

                    # Forward pass
                    y_val = model(batch)

                    # Compute val loss
                    J = loss(batch.y, y_val)

                    test_losses.append(J.cpu().numpy())

            test_loss = np.mean(test_losses)
            train_loss = np.mean(train_losses)

            wandb.log({
                "epoch": epoch + 1,
                "test_loss": test_loss,
                "train_loss": train_loss
            })

            if test_loss < best_test_loss:
                best_test_loss = test_loss

                if self.training_config.get("save_model", False):
                    checkpoint_path = self.model_checkpoint(
                        model,
                        test_loss,
                        epoch + 1
                    )

                    if wandb.run is not None:
                        wandb.run.summary["best_test_loss"] = float(
                            best_test_loss
                        )

                        wandb.run.summary["best_epoch"] = epoch + 1

                        wandb.run.summary["best_checkpoint"] = str(
                            checkpoint_path
                        )

    def train_runs(self):
        train_dl = DataLoader(
            dataset=self.train_ds, 
            batch_size=self.training_config["batch_size"], 
            shuffle=True, 
            num_workers=32
        )

        test_dl = DataLoader(
            dataset=self.test_ds,
            batch_size=self.training_config["batch_size"], 
            shuffle=True, 
            num_workers=32
        )

        lr = self.training_config["learning_rate"]
        epochs = self.training_config["num_epochs"] 

        for _ in range(1):
            run = wandb.init(project="GNN-image-gnn_train-CIFAR10", reinit=True)

            model : nn.Module = getattr(sys.modules[__name__], self.model_config["model"])
            model = model(**model.pre_init(self.model_config["args"])).to(self.device)

            loss = getattr(sys.modules[__name__], self.training_config["loss"])
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=1e-6)

            for epoch in tqdm(range(epochs)):
                test_losses = []
                train_losses = []
                train_batch_acc = []
                test_batch_acc = []

                for i, train_batch in tqdm(enumerate(train_dl)):
                    batch = train_batch.to(self.device)

                    # Forward pass
                    y_hat = model(batch)

                    # Compute loss
                    J = loss(batch.y, y_hat)
                    train_losses.append(J.detach().cpu().numpy())
                    train_batch_acc.append(100 * (sum(batch.y.detach() == torch.max(y_hat, axis=1).indices.detach()) / batch.y.detach().shape[0]).item())

                    # Backward pass
                    J.backward()

                    # Optimization step
                    optimizer.step()

                    optimizer.zero_grad()

                with torch.no_grad():
                    for i, test_batch in enumerate(test_dl):
                        batch = test_batch.to(self.device)

                        # Forward pass
                        y_val = model(batch)

                        # Compute val loss
                        J = loss(batch.y, y_val)
                        test_losses.append(J.cpu().numpy())
                        test_batch_acc.append(100 * (sum(batch.y.detach() == torch.max(y_val, axis=1).indices.detach()) / batch.y.detach().shape[0]).item())

                test_loss = np.mean(test_losses)
                train_loss = np.mean(train_losses)
                test_acc = np.mean(test_batch_acc)
                train_acc = np.mean(train_batch_acc)

                wandb.log({
                    "test_loss": test_loss,
                    "train_loss": train_loss,
                    "test_acc": test_acc,
                    "train_acc": train_acc,
                    "sample_size": self.train_ds.sample_size
                })

                scheduler.step(test_loss)

    def train_eval(self):
        train_dl = DataLoader(
            dataset=self.train_ds, 
            batch_size=self.training_config["batch_size"], 
            shuffle=True, 
            num_workers=32
        )

        test_dl = DataLoader(
            dataset=self.test_ds, 
            batch_size=self.training_config["batch_size"], 
            shuffle=True, 
            num_workers=32
        )

        num_samples = len(self.train_ds)
        batch_size = train_dl.batch_size

        lr = self.training_config["learning_rate"]
        epochs = self.training_config["num_epochs"] 
        save_model = self.training_config["save_model"]

        model : nn.Module = getattr(sys.modules[__name__], self.model_config["model"])
        model = model(**model.pre_init(self.model_config["args"])).to(self.device)
        print(type(self.train_ds))
        print(model)

        loss = getattr(sys.modules[__name__], self.training_config["loss"])
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=5e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=1e-6)

        train_losses = []
        for epoch in tqdm(range(epochs)):
            test_losses = []
            train_acc = []
            test_acc = []

            for i, train_batch in enumerate(train_dl):
                batch = train_batch.to(self.device)

                # Forward pass
                y_hat = model(batch)

                # Compute loss
                J = loss(batch.y, y_hat)
                # train_acc.append(
                #     100 * (
                #         sum(batch.y.detach() == torch.max(y_hat, axis=1).indices.detach()) / batch.y.detach().shape[0]
                #     ).item()
                # )

                # Backward pass
                J.backward()

                # Optimization step
                optimizer.step()

                optimizer.zero_grad()

                if i % 20 == 0:
                    print('Train Epoch {}/{} [{:>5}/{} ({:>2.0f}%)] | Loss: {}'.format(
                        epoch+1, epochs, i * batch_size, num_samples, 
                        100*i / len(train_dl), J.detach())
                    )
                    train_losses.append(J.detach())

            with torch.no_grad():
                for i, test_batch in enumerate(test_dl):
                    batch = test_batch.to(self.device)

                    # Forward pass
                    y_val = model(batch)

                    # Compute val loss
                    J = loss(batch.y, y_val)
                    # test_acc.append(
                    #     100 * (
                    #         sum(batch.y.detach() == torch.max(y_val, axis=1).indices.detach()) / batch.y.detach().shape[0]
                    #     ).item()
                    # )

                    test_losses.append(J.cpu().numpy())

                print(f"Test Loss: {np.mean(test_losses)}")
                # print(f"Test Acc: {np.mean(test_acc):.2f}%")
                # print(f"Train Acc: {np.mean(train_acc):.2f}%")                

            # scheduler.step(np.mean(test_losses))            

        # print(f"{acc:.2f}%")

    def train_eval_grow_graph(self):
        train_dl = self.parse_grow_graph(
            self.train_ds,
            self.training_config["batch_size"]
        )
        # num_neighbors has 2 entries (2 hops) to match config/models/gnn.yaml
        # (layers: 2). A single-entry [-1] here only materializes 1-hop
        # neighborhoods, so the model's second GCN layer aggregated over an
        # incidental, incomplete neighbor set at eval time -- a mismatch with
        # training's 2-hop sample_subgraph() sampling.
        test_dl = NeighborLoader(
            self.test_ds.data, 
            input_nodes=self.test_ds.data.mask.bool(), 
            num_neighbors=[-1, -1], 
            batch_size=512
        )

        batch_size = train_dl[0].batch_size

        lr = self.training_config["learning_rate"]
        epochs = self.training_config["num_epochs"] 
        save_model = self.training_config["save_model"]
        # Infer GNN dimensions directly from graph data
        in_features = self.train_ds.data.x.shape[1]

        out_size = (
            int(self.train_ds.data.y.max().item()) + 1
        )

        self.model_config["args"]["in_features"] = in_features
        self.model_config["args"]["out_size"] = out_size

        print(f"GNN input features: {in_features}")
        print(f"Number of classes: {out_size}")
        model : nn.Module = getattr(sys.modules[__name__], self.model_config["model"])
        model = model(**model.pre_init(self.model_config["args"])).to(self.device)
        print(type(self.train_ds))
        print(model)

        loss = getattr(sys.modules[__name__], self.training_config["loss"])
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.0)

        # Total epoch budget (config/gnn_training_config.yaml's num_epochs)
        # is now split evenly across the growth steps below, so the
        # cumulative training this function does across all steps stays
        # comparable to train_eval_gap_curve()'s fixed per-model epoch
        # budget, rather than multiplying by len(train_dl) as it would
        # if each step still got the full epoch count (the previous
        # n_increases=1 bug meant this never actually mattered before).
        n_iter_per_epoch = max(1, int(epochs // len(train_dl)))
        for i in tqdm(range(len(train_dl))):
            train_losses = []
            num_samples = train_dl[i].data.m
            print(f"\n=== Grow-graph step {i + 1}/{len(train_dl)} | n={num_samples} | {n_iter_per_epoch} epochs this step ===")

            for epoch in tqdm(range(i * n_iter_per_epoch, (i + 1) * n_iter_per_epoch)):
                test_losses = []
                train_acc = []
                test_acc = []

                model.train()
                for j, train_batch in enumerate(train_dl[i]):
                    batch = train_batch.to(self.device)

                    # Forward pass
                    y_hat = model(batch)

                    # Compute loss. GNNModel.forward() now returns only the seed
                    # nodes' rows (out[:batch.batch_size]), so the target labels
                    # are sliced the same way instead of the old batch.mask.bool()
                    # selector, which (since sample_subgraph() only keeps edges
                    # between already-sampled training nodes) was true for nearly
                    # the whole local batch -- scoring under-supported sampled
                    # neighbors right alongside the real seed-node predictions.
                    J = loss(
                        batch.y[:batch.batch_size].reshape(-1).to(torch.long), 
                        y_hat
                    )
                    train_acc.append(
                        100 * (
                            sum
                            (
                                batch.y[:batch.batch_size].reshape(-1).detach() == torch.max(y_hat, axis=1).indices.detach()
                            ) / batch.y[:batch.batch_size].reshape(-1).detach().shape[0]
                        ).item()
                    )

                    # Backward pass
                    J.backward()

                    # Optimization step
                    optimizer.step()

                    optimizer.zero_grad()

                    if j % 20 == 0:
                        print('Train Epoch {}/{} [{:>5}/{} ({:>2.0f}%)] | Loss: {}'.format(
                            epoch+1, epochs, j * batch_size, num_samples, 
                            100*j / len(train_dl[i]), J.detach())
                        )
                        train_losses.append(J.detach())

                model.eval()
                with torch.no_grad():
                    for j, test_batch in enumerate(test_dl):
                        batch = test_batch.to(self.device)

                        # Forward pass
                        y_val = model(batch)

                        # Compute val loss -- see the train-block comment above on
                        # why this now slices by :batch.batch_size instead of
                        # batch.mask.bool().
                        J = loss(
                            batch.y[:batch.batch_size].reshape(-1).to(torch.long), 
                            y_val
                        )
                        test_acc.append(
                            100 * (
                                sum
                                (
                                    batch.y[:batch.batch_size].reshape(-1).detach() == torch.max(y_val, axis=1).indices.detach()
                                ) / batch.y[:batch.batch_size].reshape(-1).detach().shape[0]
                            ).item()
                        )

                        test_losses.append(J.cpu().numpy())

                    print(f"Test Loss: {np.mean(test_losses)}")
                    print(f"Test Acc: {np.mean(test_acc):.2f}%")
                    print(f"Train Acc: {np.mean(train_acc):.2f}%")                

    def train_eval_gap_curve(
        self,
        size_fractions=(0.1, 0.25, 0.5, 0.75, 1.0),
        num_seeds=3,
        results_dir="results/gap_curve",
        reference_n=None
    ):
        """
        Generalization-gap-vs-graph-size curve (Gap 1).

        Unlike train_eval_grow_graph(), which trains ONE model
        continuously across a growing sequence of graph sizes (the base
        paper's Corollary IV.7 procedure -- a single reference point),
        this trains a FRESH, independently-initialized model from
        scratch at each requested graph size, evaluates it once on the
        same held-out test set, and discards it. That independence is
        what makes the resulting points comparable to each other and
        usable as an actual gap-vs-n curve (the base paper's Theorem
        IV.5 / Figure 2 blue curve).

        size_fractions: fractions of the dataset's eligible training
            nodes to sample at each graph size (fractions rather than
            absolute counts so the same call is meaningful across
            differently-sized datasets). See reference_n below for the
            matched-training-size variant of this.
        reference_n: if given, size_fractions become fractions of this
            fixed value instead of this dataset's own eligible-node count --
            e.g. set to the smallest dataset's own size so every dataset in
            a sweep trains on the same absolute sample sizes rather than the
            same *proportion* of its own (differently-sized) pool. None
            (default) preserves the original per-dataset-relative behavior.
            sample_subgraph() still clips n_target to this dataset's own
            n_eligible, so a reference_n larger than a given dataset's pool
            is silently capped there rather than erroring -- worth checking
            eligible counts before picking reference_n for a new dataset set.
        num_seeds: number of independent (subgraph sample + model init)
            repeats per size, for averaging / error bars. The same seed
            value is reused for both the subgraph sample and the model
            init at every size, so e.g. seed 0's node sample is nested
            across increasing sizes (a deliberate choice to reduce
            noise across the sweep, not an error). Either a single int
            (same seed count at every fraction) or a dict mapping each
            fraction to its own seed count -- smaller fractions produce
            noisier subgraph samples, so it's often worth spending more
            seeds there and fewer at large fractions that are already
            more stable.
        results_dir: directory the per-dataset results CSV is written
            to (created if missing); rows are appended, so re-running
            with more seeds or sizes does not erase earlier rows.

        Note on the recorded train_acc: it is measured in eval mode
        (dropout off, BatchNorm running stats) on the same train_dl
        subgraph, not during the training forward/backward pass -- we
        only care about the trained model's own accuracy on the data
        it was trained on, not the noisier accuracy of a mid-update,
        dropout-active network. This keeps train_acc and test_acc
        directly comparable (same eval-mode conditions, different
        data), so gen_gap_acc reflects actual generalization.
        """
        # num_neighbors has 2 entries (2 hops) to match config/models/gnn.yaml
        # (layers: 2). A single-entry [-1] here only materializes 1-hop
        # neighborhoods, so the model's second GCN layer aggregated over an
        # incidental, incomplete neighbor set at eval time -- a mismatch with
        # training's 2-hop sample_subgraph() sampling.
        test_dl = NeighborLoader(
            self.test_ds.data,
            input_nodes=self.test_ds.data.mask.bool(),
            num_neighbors=[-1, -1],
            batch_size=512
        )

        lr = self.training_config["learning_rate"]
        epochs = self.training_config["num_epochs"]
        batch_size = self.training_config["batch_size"]

        # Infer GNN dimensions directly from graph data
        in_features = self.train_ds.data.x.shape[1]

        out_size = (
            int(self.train_ds.data.y.max().item()) + 1
        )

        self.model_config["args"]["in_features"] = in_features
        self.model_config["args"]["out_size"] = out_size

        eligible_idx = torch.where(
            self.train_ds.data.mask.bool()
        )[0]

        n_eligible = eligible_idx.shape[0]

        # See reference_n's docstring above: when set, every fraction below
        # is resolved against this fixed value instead of n_eligible.
        size_base = reference_n if reference_n is not None else n_eligible

        print(f"GNN input features: {in_features}")
        print(f"Number of classes: {out_size}")
        print(f"Eligible training nodes: {n_eligible}")
        if reference_n is not None:
            print(f"Using fixed reference_n={reference_n} as the size_fractions base instead of n_eligible={n_eligible}")

        loss = getattr(sys.modules[__name__], self.training_config["loss"])

        records = []

        for fraction in size_fractions:
            n_target = max(1, round(fraction * size_base))

            # num_seeds may be a single int (same count everywhere) or a dict
            # keyed by fraction (per-fraction count) -- resolve it here so the
            # seed loop below is agnostic to which form was passed in.
            fraction_num_seeds = (
                num_seeds[fraction] if isinstance(num_seeds, dict) else num_seeds
            )

            for seed in range(fraction_num_seeds):
                print()
                print(f"=== {self.dataset_name} | fraction={fraction} | n_target={n_target} | seed={seed} ===")

                train_dl = self.sample_subgraph(
                    self.train_ds,
                    n_target,
                    batch_size,
                    seed=seed
                )

                n_actual = train_dl.data.m

                # Reseed so each size/seed combination starts from an
                # independent, but reproducible, weight initialization.
                torch.manual_seed(seed)

                model : nn.Module = getattr(sys.modules[__name__], self.model_config["model"])
                model = model(**model.pre_init(dict(self.model_config["args"]))).to(self.device)

                optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.0)

                train_acc, test_acc = [], []
                train_loss_vals, test_loss_vals = [], []

                for epoch in tqdm(range(epochs)):
                    train_loss_vals = []

                    model.train()
                    for train_batch in train_dl:
                        batch = train_batch.to(self.device)

                        # Forward pass
                        y_hat = model(batch)

                        # Compute loss. GNNModel.forward() now returns only the seed
                        # nodes' rows (out[:batch.batch_size]), so the target labels
                        # are sliced the same way instead of the old batch.mask.bool()
                        # selector, which (since sample_subgraph() only keeps edges
                        # between already-sampled training nodes) was true for nearly
                        # the whole local batch -- scoring under-supported sampled
                        # neighbors right alongside the real seed-node predictions.
                        J = loss(
                            batch.y[:batch.batch_size].reshape(-1).to(torch.long),
                            y_hat
                        )

                        train_loss_vals.append(J.detach().cpu().numpy())

                        # Backward pass
                        J.backward()

                        # Optimization step
                        optimizer.step()

                        optimizer.zero_grad()

                    # train_acc is NOT taken from the training loop above -- that
                    # forward pass has dropout randomly zeroing units and BatchNorm
                    # using each mini-batch's own statistics mid-optimizer-update, so
                    # its accuracy reflects a noisy, still-changing network rather
                    # than the trained model itself. Instead, train_acc is measured
                    # here, in eval mode on the same train_dl subgraph, right
                    # alongside test_acc (also eval-mode) -- so both numbers reflect
                    # the same finished, clean model and gen_gap_acc captures actual
                    # generalization rather than a train-mode-vs-eval-mode artifact.
                    train_acc = []
                    test_acc = []
                    test_loss_vals = []

                    model.eval()
                    with torch.no_grad():
                        for train_batch in train_dl:
                            batch = train_batch.to(self.device)

                            y_hat = model(batch)

                            train_acc.append(
                                100 * (
                                    sum(
                                        batch.y[:batch.batch_size].reshape(-1).detach() == torch.max(y_hat, axis=1).indices.detach()
                                    ) / batch.y[:batch.batch_size].reshape(-1).detach().shape[0]
                                ).item()
                            )

                        for test_batch in test_dl:
                            batch = test_batch.to(self.device)

                            # Forward pass
                            y_val = model(batch)

                            # Compute val loss -- see the train-block comment above
                            # on why this now slices by :batch.batch_size instead of
                            # batch.mask.bool().
                            J = loss(
                                batch.y[:batch.batch_size].reshape(-1).to(torch.long),
                                y_val
                            )

                            test_acc.append(
                                100 * (
                                    sum(
                                        batch.y[:batch.batch_size].reshape(-1).detach() == torch.max(y_val, axis=1).indices.detach()
                                    ) / batch.y[:batch.batch_size].reshape(-1).detach().shape[0]
                                ).item()
                            )

                            test_loss_vals.append(J.cpu().numpy())

                    print(
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"Train Acc: {np.mean(train_acc):.2f}% | "
                        f"Test Acc: {np.mean(test_acc):.2f}% | "
                        f"Train Loss: {np.mean(train_loss_vals):.4f} | "
                        f"Test Loss: {np.mean(test_loss_vals):.4f}"
                    )

                records.append({
                    "dataset": self.dataset_name,
                    "fraction": fraction,
                    "reference_n": size_base,
                    "n_target": n_target,
                    "n_actual": n_actual,
                    "seed": seed,
                    "epochs": epochs,
                    "train_acc": float(np.mean(train_acc)),
                    "test_acc": float(np.mean(test_acc)),
                    "gen_gap_acc": float(np.mean(train_acc) - np.mean(test_acc)),
                    "train_loss": float(np.mean(train_loss_vals)),
                    "test_loss": float(np.mean(test_loss_vals)),
                    "gen_gap_loss": float(np.mean(test_loss_vals) - np.mean(train_loss_vals)),
                    "timestamp": datetime.now().isoformat()
                })

        self.write_gap_curve_results(records, results_dir)

        return records

    def write_gap_curve_results(self, records, results_dir):
        if not records:
            return None

        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        results_path = results_dir / f"{self.dataset_name}.csv"

        fieldnames = list(records[0].keys())
        write_header = not results_path.exists()

        with open(results_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            writer.writerows(records)

        print(f"Wrote {len(records)} rows to {results_path}")

        return results_path

    def train_eval_mlp_baseline(
        self,
        results_dir="results/mlp_baseline",
        n_target=None,
        num_seeds=1
    ):
        """
        Non-graph baseline for Gap 2: trains an MLP directly on the same
        VAE embeddings the GNN consumes (Step 5's plain per-dataset
        embedding file), with no KNN graph and no neighbor sampling --
        just an ordinary DataLoader over independent samples. Comparing
        this against the GNN's result (Step 11 / Step 11a) is what
        separates "the graph structure helps" from "this dataset is
        just easy to classify from its embedding alone".

        in_features and out_size are inferred from the loaded data at
        runtime rather than read from the model config, since a
        dataset's actual VAE latent size depends on which sweep trial
        won and isn't safe to hardcode per dataset.

        n_target: if given, each restart trains on a random n_target-row
            subsample of self.train_ds (drawn with that restart's own seed,
            capped at however many rows this dataset actually has) instead
            of every row -- for comparing against a GNN gap-curve point
            trained at the same matched size. None (default) preserves the
            original behavior: train on the full training set every time.
        num_seeds: number of independent (subsample draw + model init)
            restarts, appended as separate rows. Default 1 matches the
            original single-run behavior -- note the original single run
            had no explicit seed at all, so this makes even that default
            case reproducible (seed 0) where it previously was not.
        """
        test_dl = DataLoader(
            dataset=self.test_ds,
            batch_size=self.training_config["batch_size"],
            shuffle=False
        )

        lr = self.training_config["learning_rate"]
        epochs = self.training_config["num_epochs"]

        in_features = self.train_ds.x.shape[1]
        out_size = int(self.train_ds.targets.max().item()) + 1

        self.model_config["args"]["in_features"] = in_features
        self.model_config["args"]["out_size"] = out_size

        print(f"MLP input features: {in_features}")
        print(f"Number of classes: {out_size}")

        loss = getattr(sys.modules[__name__], self.training_config["loss"])

        n_available = len(self.train_ds)

        records = []

        for seed in range(num_seeds):
            if n_target is not None:
                n_use = min(n_target, n_available)
                generator = torch.Generator().manual_seed(seed)
                perm = torch.randperm(n_available, generator=generator)[:n_use]
                train_subset = Subset(self.train_ds, perm.tolist())
            else:
                n_use = n_available
                train_subset = self.train_ds

            train_dl = DataLoader(
                dataset=train_subset,
                batch_size=self.training_config["batch_size"],
                shuffle=True
            )

            print()
            print(f"=== {self.dataset_name} | n_target={n_target} | n_actual={n_use} | seed={seed} ===")

            # Reseed so each restart starts from an independent, but
            # reproducible, weight initialization -- same convention as
            # train_eval_gap_curve()'s per-(size, seed) reseeding.
            torch.manual_seed(seed)

            model : nn.Module = getattr(sys.modules[__name__], self.model_config["model"])
            model = model(**model.pre_init(dict(self.model_config["args"]))).to(self.device)

            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.0)

            train_acc, test_acc = [], []
            train_loss_vals, test_loss_vals = [], []

            for epoch in tqdm(range(epochs)):
                train_acc = []
                train_loss_vals = []

                model.train()
                for train_batch in train_dl:
                    batch = train_batch.to(self.device)

                    # Forward pass
                    y_hat = model(batch)

                    # Compute loss
                    J = loss(
                        batch.y[batch.mask.bool()].reshape(-1).to(torch.long),
                        y_hat
                    )

                    train_acc.append(
                        100 * (
                            sum(
                                batch.y[batch.mask.bool()].reshape(-1).detach() == torch.max(y_hat, axis=1).indices.detach()
                            ) / batch.y[batch.mask.bool()].reshape(-1).detach().shape[0]
                        ).item()
                    )

                    train_loss_vals.append(J.detach().cpu().numpy())

                    # Backward pass
                    J.backward()

                    # Optimization step
                    optimizer.step()

                    optimizer.zero_grad()

                test_acc = []
                test_loss_vals = []

                model.eval()
                with torch.no_grad():
                    for test_batch in test_dl:
                        batch = test_batch.to(self.device)

                        # Forward pass
                        y_val = model(batch)

                        # Compute val loss
                        J = loss(
                            batch.y[batch.mask.bool()].reshape(-1).to(torch.long),
                            y_val
                        )

                        test_acc.append(
                            100 * (
                                sum(
                                    batch.y[batch.mask.bool()].reshape(-1).detach() == torch.max(y_val, axis=1).indices.detach()
                                ) / batch.y[batch.mask.bool()].reshape(-1).detach().shape[0]
                            ).item()
                        )

                        test_loss_vals.append(J.cpu().numpy())

                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Acc: {np.mean(train_acc):.2f}% | "
                    f"Test Acc: {np.mean(test_acc):.2f}% | "
                    f"Train Loss: {np.mean(train_loss_vals):.4f} | "
                    f"Test Loss: {np.mean(test_loss_vals):.4f}"
                )

            records.append({
                "dataset": self.dataset_name,
                "n_target": n_target,
                "n_actual": n_use,
                "seed": seed,
                "epochs": epochs,
                "train_acc": float(np.mean(train_acc)),
                "test_acc": float(np.mean(test_acc)),
                "gen_gap_acc": float(np.mean(train_acc) - np.mean(test_acc)),
                "train_loss": float(np.mean(train_loss_vals)),
                "test_loss": float(np.mean(test_loss_vals)),
                "gen_gap_loss": float(np.mean(test_loss_vals) - np.mean(train_loss_vals)),
                "timestamp": datetime.now().isoformat()
            })

        self.write_mlp_baseline_results(records, results_dir)

        return records

    def write_mlp_baseline_results(self, records, results_dir):
        if not records:
            return None

        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        results_path = results_dir / f"{self.dataset_name}.csv"

        fieldnames = list(records[0].keys())
        write_header = not results_path.exists()

        with open(results_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            writer.writerows(records)

        print(f"Wrote {len(records)} row(s) to {results_path}")

        return results_path
