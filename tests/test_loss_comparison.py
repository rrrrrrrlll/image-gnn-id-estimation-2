"""Numerical and protocol checks; run with python -m unittest discover -s tests."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/scripts"))
import torch
from torch_geometric.data import Data
from training.id_loss import local_log_id
from models.classifier import GNNModel
from compare_losses import training_graph


class IDTests(unittest.TestCase):
    def test_known_mom_values(self):
        x = torch.tensor([[0.], [1.], [3.]])
        # k=2: d_hat = nearest / (second_nearest - nearest).
        torch.testing.assert_close(local_log_id(x, 2).exp(), torch.tensor([.5, 1., 2.]))

    def test_gradients_and_scale(self):
        torch.manual_seed(7)
        x = torch.randn(30, 5, requires_grad=True)
        value = local_log_id(x, 8)
        torch.testing.assert_close(value, local_log_id(x * 7, 8))
        (-value.mean()).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_duplicates_are_finite_and_collapse_is_low(self):
        x = torch.zeros(8, 4, requires_grad=True)
        loss = -local_log_id(x).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(loss.item(), 10)

    def test_train_graph_excludes_test_nodes(self):
        graph = Data(x=torch.randn(5, 3), y=torch.arange(5),
                     edge_index=torch.tensor([[0, 1, 2, 3], [1, 4, 3, 4]]),
                     edge_weight=torch.ones(4))
        graph.n_train = 3
        result = training_graph(graph)
        self.assertEqual(result.num_nodes, 3)
        torch.testing.assert_close(result.edge_index, torch.tensor([[0], [1]]))

    def test_feature_return_preserves_logits_and_id_updates_gnn(self):
        torch.manual_seed(3)
        model = GNNModel(in_features=5, hidden_size=16, out_size=3, gnn_conv="GCNConv",
                         gnn_conv_args={}, layers=2, dropout=.5)
        batch = Data(x=torch.randn(10, 5), edge_index=torch.empty(2, 0, dtype=torch.long),
                     edge_weight=torch.empty(0), batch_size=8)
        model.eval()
        plain = model(batch)
        logits, features = model(batch, return_embeddings=True)
        torch.testing.assert_close(plain, logits)
        self.assertEqual(tuple(features.shape), (8, 16))
        (-local_log_id(features, 4).mean()).backward()
        grad = model.layers[0].layers[0].lin.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
