import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from config import *
from collections import OrderedDict
from torch_geometric.data import Batch
from .graph_utils import solve_num_nodes, tensor_to_batch_graph
from torch_geometric.utils import to_dense_adj

class GCNBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
                 activation_cls: nn.Module= nn.ReLU, dropout: float=0.1, ratio: float=0.5):
        super(GCNBlock, self).__init__()
        self.conv = gnn.GCNConv(input_dim, output_dim)
        self.nonlinear = activation_cls()
        self.bn = gnn.BatchNorm(output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if ratio < 0.99:
            self.pool = gnn.TopKPooling(output_dim, ratio=ratio)
        
    def forward(self, graphs: Batch):
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

class GCN(nn.Module):
    def __init__(self, config: GCNConfig):
        super(GCN, self).__init__()
        layers_dict = OrderedDict()
        self.num_nodes = solve_num_nodes(config.input_dim)
        self.output_dim = config.hidden_dims[-1]
        last_hidden_dim = 1
        last_num_nodes = self.num_nodes
        for i, hidden_dim in enumerate(config.hidden_dims):
            layers_dict[f'gcnblock_{i}'] = GCNBlock(
                last_hidden_dim, 
                hidden_dim, 
                dropout=config.dropout,
                ratio=config.num_nodes[i]/last_num_nodes
                )
            last_hidden_dim = hidden_dim
            last_num_nodes = config.num_nodes[i]
        self.gcn_layers = nn.Sequential(layers_dict)
        # self.pool = nn.Linear(last_num_nodes * self.output_dim, self.output_dim)
        self.pool = gnn.global_max_pool
        
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
        graphs: Batch = self.gcn_layers(graphs)
        # features = graphs.x.view(batch_size, -1) # (batch_size, last_num_nodes * output_dim)
        # features = self.pool(features) # (batch_size, output_dim)
        features = self.pool(graphs.x, graphs.batch)
        return features