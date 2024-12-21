import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from config import *
from collections import OrderedDict
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

def tensor_to_batch_graph(flattened_tensor, num_nodes=400):
    """
    Convert a flattened tensor to a batch graph.
    
    Args:
    """
    batch_size, dim_feats = flattened_tensor.shape
    assert dim_feats == (num_nodes * (num_nodes - 1)) // 2, \
        "The flattened tensor does not match the number of nodes."

    triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1) # not including the diagonal
    
    # Construct the adjacency matrices
    adj_matrices = torch.zeros((batch_size, num_nodes, num_nodes))
    adj_matrices[:, triu_indices[0], triu_indices[1]] = flattened_tensor
    adj_matrices = adj_matrices + adj_matrices.transpose(-1, -2) # symmetry

    # Construct the batch graph
    graphs = []
    for i in range(batch_size):
        edge_index, edge_attr = dense_to_sparse(adj_matrices[i])
        node_features = torch.ones((num_nodes, 1))
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(graph)

    batch_graph = Batch.from_data_list(graphs)
    return batch_graph

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
        last_hidden_dim = config.input_dim
        for i, hidden_dim in enumerate(config.hidden_dims):
            layers_dict[f'gcnblock_{i}'] = GCNBlock(last_hidden_dim, hidden_dim, dropout=config.dropout)
            last_hidden_dim = hidden_dim
        self.gcn_layers = nn.Sequential(layers_dict)
        
    def forward(self, data: torch.Tensor, attention_mask: torch.Tensor=None):
        """_summary_

        Args:
            data (torch.Tensor): (batch_size, input_dim)
            attention_mask (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        graphs = tensor_to_batch_graph(data)
        features = self.gcn_layers(graphs)
        return features