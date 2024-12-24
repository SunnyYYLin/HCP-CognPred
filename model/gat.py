import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from config import *
from collections import OrderedDict
from torch_geometric.data import Batch
from .graph_utils import solve_num_nodes, tensor_to_batch_graph
from torch_geometric.utils import to_dense_adj
from dataclasses import dataclass, field

class GATBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int,
                 activation_cls: nn.Module = nn.ReLU, dropout: float = 0.0, ratio: float = 0.5):
        """
        A single Graph Attention Network (GAT) block.

        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output features per head.
            num_heads (int): Number of attention heads.
            activation_cls (nn.Module, optional): Activation function class. Defaults to nn.ReLU.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            ratio (float, optional): Pooling ratio. Defaults to 0.5.
        """
        super(GATBlock, self).__init__()
        self.conv = gnn.GATConv(input_dim, output_dim, heads=num_heads, dropout=dropout, concat=True)
        self.nonlinear = activation_cls()
        self.bn = gnn.BatchNorm(output_dim * num_heads)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if ratio < 0.99:
            self.pool = gnn.TopKPooling(output_dim * num_heads, ratio=ratio)
        
    def forward(self, graphs: Batch):
        """
        Forward pass of the GAT block.

        Args:
            graphs (Batch): Batch of input graphs.

        Returns:
            Batch: Batch of output graphs after applying GAT and pooling.
        """
        features = self.conv(graphs.x, graphs.edge_index, graphs.edge_attr)
        features = self.nonlinear(features)
        features = self.bn(features)
        features = self.dropout(features)
        if hasattr(self, 'pool'):
            output = self.pool(features, graphs.edge_index, graphs.edge_attr, graphs.batch)
            x, edge_index, edge_attr, batch, _, _ = output
            next_graphs = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        else:
            next_graphs = Batch(x=features, edge_index=graphs.edge_index, edge_attr=graphs.edge_attr, batch=graphs.batch)
        return next_graphs

class GAT(nn.Module):
    def __init__(self, config: 'GATConfig'):
        """
        Graph Attention Network (GAT) model composed of multiple GAT blocks.

        Args:
            config (GATConfig): Configuration object containing hyperparameters.
        """
        super(GAT, self).__init__()
        layers_dict = OrderedDict()
        self.num_nodes = solve_num_nodes(config.input_dim)
        self.output_dim = config.hidden_dims[-1] * config.num_heads[-1]
        last_hidden_dim = 1
        last_num_nodes = self.num_nodes

        for i, (hidden_dim, num_heads) in enumerate(zip(config.hidden_dims, config.num_heads)):
            layers_dict[f'gatblock_{i}'] = GATBlock(
                input_dim=last_hidden_dim,
                output_dim=hidden_dim,
                num_heads=num_heads,
                dropout=config.dropout,
                ratio=config.num_nodes[i] / last_num_nodes
            )
            last_hidden_dim = hidden_dim * num_heads  # Update for the next layer
            last_num_nodes = config.num_nodes[i]

        self.gat_layers = nn.Sequential(layers_dict)
        self.pool = lambda x, batch: torch.cat([
            gnn.global_mean_pool(x, batch),
            gnn.global_max_pool(x, batch)
        ], dim=-1)
        
    def forward(self, data: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Forward pass of the GAT model.

        Args:
            data (torch.Tensor): Input feature tensor of shape (batch_size, input_dim).
            attention_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            torch.Tensor: Output features after GAT layers and pooling.
        """
        batch_size, dim_feats = data.shape
        graphs = tensor_to_batch_graph(data)
        graphs: Batch = self.gat_layers(graphs)
        features = self.pool(graphs.x, graphs.batch)
        return features