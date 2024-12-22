import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

def solve_num_nodes(dim_feats: int) -> int:
    """num_nodes * (num_nodes - 1) / 2 = dim_feats
    """
    num_nodes = int((1 + (1 + 8 * dim_feats) ** 0.5) / 2)
    assert num_nodes * (num_nodes - 1) == 2 * dim_feats, \
        "The number of nodes is not an integer."
    return num_nodes

def tensor_to_batch_graph(flattened_tensor: torch.Tensor, num_nodes=400) -> Batch:
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