import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from config import *
from collections import OrderedDict

class GCNBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float=0.1):
        super(GCNBlock, self).__init__()
        self.conv = GCNConv(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class GCN(nn.Module):
    def __init__(self, config: GCNConfig):
        super(GCN, self).__init__()
        layers_dict = OrderedDict()
        last_hidden_dim = config.input_dim
        for i, hidden_dim in enumerate(config.hidden_dims):
            layers_dict[f'gcnblock_{i}'] = GCNBlock(last_hidden_dim, hidden_dim, dropout=config.dropout)
            last_hidden_dim = hidden_dim
        self.gcn_layers = nn.Sequential(layers_dict)
        self.output_layer = GCNConv(last_hidden_dim, config.output_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
        x = self.output_layer(x, edge_index)
        return x

