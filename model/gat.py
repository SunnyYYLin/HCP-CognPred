import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from config import *
from collections import OrderedDict
from .graph_utils import solve_num_nodes, tensor_to_batch_graph

class GATBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
                 num_heads: int, activation_cls: nn.Module = nn.ReLU, dropout: float = 0.1):
        super(GATBlock, self).__init__()
        self.conv = gnn.GATConv(input_dim, output_dim, heads=num_heads, dropout=dropout, concat=True)
        self.activation = activation_cls()
        # GATConv outputs features of size output_dim * num_heads if concat=True
        self.bn = gnn.BatchNorm(output_dim * num_heads)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv(x, edge_index)
        x = self.activation(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class GAT(nn.Module):
    def __init__(self, config: GATConfig):
        super(GAT, self).__init__()
        layers_dict = OrderedDict()
        self.num_nodes = solve_num_nodes(config.input_dim)
        self.output_dim = config.hidden_dims[-1]
        last_hidden_dim = 1
        for i, (hidden_dim, num_heads) in enumerate(zip(config.hidden_dims, config.num_heads)):
            layers_dict[f'gatblock_{i}'] = GATBlock(
                input_dim=last_hidden_dim, 
                output_dim=hidden_dim, 
                num_heads=num_heads, 
                dropout=config.dropout
            )
            last_hidden_dim = hidden_dim * num_heads
        self.gat_layers = nn.Sequential(layers_dict)
        self.pool = nn.Sequential(
            nn.Linear(last_hidden_dim * self.num_nodes, self.output_dim),
            nn.ReLU()
        )
        
    def forward(self, data: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            data (torch.Tensor): (batch_size, input_dim)
            attention_mask (torch.Tensor): Not used in this implementation
        
        Returns:
            torch.Tensor: (batch_size, output_dim)
        """
        batch_size, dim_feats = data.shape
        graphs = tensor_to_batch_graph(data)
        x, edge_index = graphs.x, graphs.edge_index
        for layer in self.gat_layers:
            x = layer(x, edge_index)
        # Assuming all graphs have the same number of nodes
        features = x.reshape(batch_size, -1)  # (batch_size, hidden_dim*num_heads*num_nodes)
        features = self.pool(features)  # (batch_size, output_dim)
        return features