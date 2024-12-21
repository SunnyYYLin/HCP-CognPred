import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from config import *
from collections import OrderedDict
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

def solve_num_nodes(dim_feats: int) -> int:
    """num_nodes * (num_nodes - 1) / 2 = dim_feats
    """
    num_nodes = int((1 + (1 + 8 * dim_feats) ** 0.5) / 2)
    assert num_nodes * (num_nodes - 1) == 2 * dim_feats, \
        "The number of nodes is not an integer."
    return num_nodes

def tensor_to_batch_graph(flattened_tensor: torch.Tensor, num_nodes=400):
    """
    Convert a flattened tensor to a batch graph.
    
    Args:
    """
    batch_size, dim_feats = flattened_tensor.shape
    device = flattened_tensor.device
    assert dim_feats == (num_nodes * (num_nodes - 1)) // 2, \
        "The flattened tensor does not match the number of nodes."

    triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device) # not including the diagonal
    
    # Construct the adjacency matrices
    adj_matrices = torch.zeros((batch_size, num_nodes, num_nodes), device=device)
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
    return batch_graph.to(device)

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
            nn.Linear(self.output_dim * self.num_nodes, self.output_dim),
            nn.ReLU(),
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
        features = x.reshape(batch_size, -1) # (batch_size, num_nodes * hidden_dim)
        features = self.pool(features) # (batch_size, hidden_dim)
        return features