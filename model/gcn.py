import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from config import *
from collections import OrderedDict
from .graph_utils import solve_num_nodes, tensor_to_batch_graph

class GCNBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
                 activation_cls: nn.Module= nn.ReLU, dropout: float=0.1):
        super(GCNBlock, self).__init__()
        self.conv = gnn.GCNConv(input_dim, output_dim)
        self.relu = activation_cls()
        self.bn = gnn.BatchNorm(output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, data: torch.Tensor, edge_index: torch.Tensor):
        data = self.conv(data, edge_index)
        data = self.relu(data)
        data = self.bn(data)
        data = self.dropout(data)
        return data
    
class GCN(nn.Module):
    def __init__(self, config: GCNConfig):
        super(GCN, self).__init__()
        layers_dict = OrderedDict()
        self.num_nodes = solve_num_nodes(config.input_dim)
        self.output_dim = config.hidden_dims[-1]
        last_hidden_dim = 1
        for i, hidden_dim in enumerate(config.hidden_dims):
            layers_dict[f'gcnblock_{i}'] = GCNBlock(last_hidden_dim, hidden_dim, dropout=config.dropout)
            last_hidden_dim = hidden_dim
        self.gcn_layers = nn.Sequential(layers_dict)
        self.pool = nn.Sequential(
            nn.LazyLinear(self.output_dim),
            nn.ReLU()
        )
        
    def forward(self, data: torch.Tensor, attention_mask: torch.Tensor=None):
        """_summary_

        Args:
            data (torch.Tensor): (batch_size, input_dim)
            attention_mask (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        batch_size, dim_feats = data.shape
        graphs = tensor_to_batch_graph(data)
        x, edge_index = graphs.x, graphs.edge_index
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
        features = x.reshape(batch_size, -1) # (batch_size, hidden_dim*num_nodes)
        features = self.pool(features) # (batch_size, hidden_dim)
        return features