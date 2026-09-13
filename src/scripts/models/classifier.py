"""Repository GNN classifier, isolated from optional VAE dependencies."""
import sys
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GATConv, GCNConv, GatedGraphConv, ARMAConv
from models.gnns import GNNBasicBlock
from models.encoders import WeightedSAGEConv


class GNNModel(nn.Module):
    """
    Graph Neural Network (GNN).

    Args:
    """

    @staticmethod
    def pre_init(config, **kwargs):
        return config

    def __init__(
        self,
        in_features,
        hidden_size,
        out_size,
        gnn_conv,
        gnn_conv_args,
        layers=1,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()

        gnn_conv = getattr(sys.modules[__name__], gnn_conv)
        self.gnn_conv = gnn_conv
        self.layers = nn.ModuleList(
            [
                # nn.Linear(in_features, hidden_size),
                GNNBasicBlock(
                    in_features, 
                    hidden_size, 
                    gnn_conv, 
                    gnn_conv_args,
                    res_connect=False
                ),
                *[
                    GNNBasicBlock(
                        hidden_size,
                        hidden_size, 
                        gnn_conv, 
                        gnn_conv_args,
                        res_connect=True
                    ) for l in range(layers-1)
                ],
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, out_size)
            ]
        )

    def _step(self):
        pass

    def _eval(self):
        pass

    def forward(self, batch, return_embeddings=False):
        out = batch.x

        if self.gnn_conv in (GATConv, GATv2Conv):
            conv_fwd_args = {
                "return_attention_weights": None,
                "edge_attr": batch.edge_weight
            }
        elif self.gnn_conv in (GCNConv, GatedGraphConv, ARMAConv):
            conv_fwd_args = {
                "edge_weight": batch.edge_weight
            }
        else:
            conv_fwd_args = {}

        embeddings = None
        for layer in self.layers:
            if isinstance(layer, GNNBasicBlock):
                out, _ = layer(out, batch.edge_index, **conv_fwd_args)
                embeddings = out
            elif isinstance(layer, self.gnn_conv):
                out = layer(out, batch.edge_index, **conv_fwd_args)
            else:
                out = layer(out)

        # NeighborLoader batches put the seed nodes -- the ones this batch is
        # actually supposed to predict -- as the first batch.batch_size rows, with
        # any sampled 1-/2-hop neighbors afterward (pulled in only to support
        # message-passing into the seeds, not meant to be scored themselves).
        # This used to slice by batch.mask instead, a leftover whole-graph
        # train/test flag: sample_subgraph() only keeps edges between
        # already-sampled training nodes, so that mask was true for nearly the
        # entire local batch, not just the seeds -- scoring predictions for
        # under-supported sampled neighbors (whose own 2-hop neighborhoods were
        # never fully expanded) right alongside the real seed-node predictions.
        logits = out[:batch.batch_size]
        if return_embeddings:
            return logits, embeddings[:batch.batch_size]
        return logits

