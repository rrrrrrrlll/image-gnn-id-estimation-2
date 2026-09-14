"""Estimator formula, gradient, loss-isolation and numerical-stability checks."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import json
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/scripts"))
import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch_geometric.data import Data

from compare_id_losses import ARMS, evaluate, objective, run_dataset
from models.classifier import GNNModel
from training.id_regularizers import METHODS, id_penalty


class RegularizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.z = torch.randn(64, 5, dtype=torch.float64)
        self.args = SimpleNamespace(lambda_id=.01, mle_k=20, twonn_discard=.1,
                                    corr_k1=10, corr_k2=20, corr_temperature=.1)

    def test_mle_matches_harmonic_formula(self):
        distances = np.sort(cdist(self.z.numpy(), self.z.numpy()), axis=1)[:, 1:21]
        local = 19 / np.log(distances[:, -1:] / distances).sum(axis=1)
        expected = np.log(1 / np.mean(1 / local))
        self.assertAlmostEqual(id_penalty(self.z, "mle").item(), expected, places=10)

    def test_twonn_matches_trimmed_regression(self):
        distances = np.sort(cdist(self.z.numpy(), self.z.numpy()), axis=1)[:, 1:3]
        n = len(self.z)
        x = np.sort(np.log(distances[:, 1] / distances[:, 0]))[:int(.9 * n)]
        y = -np.log1p(-np.arange(len(x)) / n)
        slope = np.linalg.lstsq(x[:, None], y, rcond=None)[0][0]
        self.assertAlmostEqual(id_penalty(self.z, "twonn").item(), np.log(slope), places=10)

    def test_correlation_converges_to_hard_counts(self):
        distances = cdist(self.z.numpy(), self.z.numpy())
        radii = np.sort(distances, axis=1)[:, 1:21]
        r1, r2 = np.median(radii[:, 9]), np.median(radii[:, 19])
        pairs = distances[~np.eye(len(distances), dtype=bool)]
        dimension = np.log(np.mean(pairs < r2) / np.mean(pairs < r1)) / np.log(r2 / r1)
        value = id_penalty(self.z, "corrint", corr_temperature=1e-7)
        self.assertAlmostEqual(value.item(), np.log(dimension), places=5)

    def test_finite_nonzero_gradients_and_scale_invariance(self):
        for method in METHODS:
            with self.subTest(method=method):
                z = self.z.clone().requires_grad_()
                penalty = id_penalty(z, method)
                torch.testing.assert_close(penalty, id_penalty(z * 7, method))
                penalty.backward()
                self.assertTrue(torch.isfinite(z.grad).all())
                self.assertGreater(z.grad.abs().sum().item(), 0.)

    def test_duplicates_collapse_and_small_batches(self):
        for method in METHODS:
            with self.subTest(method=method):
                duplicate = torch.cat((self.z, self.z[:8])).requires_grad_()
                torch.testing.assert_close(id_penalty(duplicate, method), id_penalty(self.z, method))
                for n in (1, 2, 3):
                    z = torch.zeros(n, 5, requires_grad=True)
                    value = id_penalty(z, method)
                    self.assertAlmostEqual(value.item(), math.log(1e-6), places=5)
                    value.backward()
                    self.assertTrue(torch.isfinite(z.grad).all())
                for n in (3, 4, 12):
                    z = self.z[:n].clone().requires_grad_()
                    value = id_penalty(z, method)
                    value.backward()
                    self.assertTrue(torch.isfinite(value))
                    self.assertTrue(torch.isfinite(z.grad).all())

    def test_each_penalty_updates_gnn(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = GNNModel(in_features=5, hidden_size=16, out_size=3,
                                 gnn_conv="GCNConv", gnn_conv_args={}, layers=2, dropout=.5)
                model.eval()
                batch = Data(x=self.z.float(), edge_index=torch.empty(2, 0, dtype=torch.long),
                             edge_weight=torch.empty(0), batch_size=len(self.z))
                _, features = model(batch, return_embeddings=True)
                id_penalty(features, method).backward()
                grad = model.layers[0].layers[0].lin.weight.grad
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0.)

    def test_zero_lambda_matches_ce_gradients(self):
        self.args.lambda_id = 0
        reference = None
        for arm in ARMS:
            z = self.z.clone().requires_grad_()
            loss, penalty = objective(z[:, :3], z, torch.arange(len(z)) % 3, arm, self.args)
            loss.backward()
            if reference is None:
                reference = (loss.detach(), z.grad.clone())
            torch.testing.assert_close(loss.detach(), reference[0], rtol=0, atol=0)
            torch.testing.assert_close(z.grad, reference[1], rtol=0, atol=0)
            self.assertEqual(penalty.item(), 0.)

    def test_total_loss_encourages_lower_estimates(self):
        # Hold classification fixed to isolate the actual regularizer's effect.
        # A small gradient step on the TOTAL objective must lower log(D), not
        # just produce nonzero gradients (which would miss a reversed sign).
        logits = torch.zeros(len(self.z), 3, dtype=self.z.dtype)
        labels = torch.arange(len(self.z)) % 3
        for arm in METHODS:
            with self.subTest(arm=arm):
                z = self.z.clone().requires_grad_()
                loss, penalty = objective(logits, z, labels, arm, self.args)
                torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits, labels)
                                           + self.args.lambda_id * penalty)
                loss.backward()
                updated = z.detach() - 1e-3 * z.grad
                self.assertLess(id_penalty(updated, arm).item(), penalty.item())

    def test_old_direction_manifest_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp), data_root=Path(tmp), limit=0, resume=True)
            arrays = [np.zeros((4, 4)), np.zeros((4, 4))]
            with patch("compare_id_losses.read_embeddings", return_value=(arrays, [])), \
                 patch("compare_id_losses.build_graph", side_effect=RuntimeError("test stops before graph")):
                with self.assertRaisesRegex(RuntimeError, "test stops before graph"):
                    run_dataset("mnist", args, {}, {})
            path = Path(tmp) / "mnist/manifest.json"
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["id_direction"], "lower")
            manifest.pop("id_direction")
            manifest.pop("objective")
            manifest["version"] = 1
            path.write_text(json.dumps(manifest))
            with patch("compare_id_losses.read_embeddings", return_value=(arrays, [])), \
                 patch("compare_id_losses.build_graph") as build:
                with self.assertRaisesRegex(ValueError, "Changed data/settings"):
                    run_dataset("mnist", args, {}, {})
                build.assert_not_called()

    def test_evaluation_never_computes_id(self):
        model = GNNModel(in_features=5, hidden_size=16, out_size=3,
                         gnn_conv="GCNConv", gnn_conv_args={}, layers=2, dropout=.5)
        graph = Data(x=self.z.float(), y=torch.arange(len(self.z)) % 3,
                     edge_index=torch.empty(2, 0, dtype=torch.long), edge_weight=torch.empty(0))
        args = SimpleNamespace(full_batch=True)
        with patch("compare_id_losses.id_penalty", side_effect=AssertionError("ID evaluation forbidden")):
            result = evaluate(model, graph, torch.arange(len(self.z)), args, 2, "cpu")
        self.assertEqual(set(result), {"ce", "accuracy"})


if __name__ == "__main__":
    unittest.main()
