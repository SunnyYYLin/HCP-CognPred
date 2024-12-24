# model/gcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, Batch
from collections import OrderedDict
from .graph_utils import solve_num_nodes, tensor_to_batch_graph
from torch_geometric.utils import to_dense_adj

# 定义 RaGConv
class RaGConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_regions, num_communities):
        super(RaGConv, self).__init__(aggr='add')  # 聚合操作：加和
        self.num_regions = num_regions
        self.num_communities = num_communities

        # 基础权重，用于计算不同区域的卷积核权重
        self.region_embedding = nn.Embedding(num_regions, num_communities)
        self.basis_weights = nn.Parameter(torch.randn(num_communities, in_channels, out_channels))

        # 可学习的消息传递权重
        self.edge_weight = nn.Linear(1, 1)

    def forward(self, x, edge_index, edge_attr, region_indices):
        assert x.size(0) == region_indices.size(0), "节点数和区域索引数必须匹配"
        
        # region_indices: 每个节点对应的区域索引
        region_embed = self.region_embedding(region_indices)  # (num_nodes, num_communities)
        
        # 计算每个节点的卷积核权重
        kernel_weights = torch.einsum('nc,cio->nio', region_embed, self.basis_weights)  # (num_nodes, in_channels, out_channels)

        # 传递消息
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, kernel_weights=kernel_weights)
        return F.relu(out)  # 添加非线性激活

    def message(self, x_j, edge_attr, kernel_weights):
        # 消息传递，结合边权重（edge_attr）和卷积核权重
        edge_attr = self.edge_weight(edge_attr).sigmoid()
        return torch.einsum('nij,nj->ni', kernel_weights, x_j) * edge_attr

# 定义 RPool
class RPool(nn.Module):
    def __init__(self, in_channels, pool_ratio=0.5):
        super(RPool, self).__init__()
        self.score_projection = nn.Linear(in_channels, 1)  # 投影向量
        self.pool_ratio = pool_ratio

    def forward(self, x, edge_index, edge_attr, region_indices, batch):
        # 计算节点得分
        scores = self.score_projection(x).squeeze()  # (num_nodes, )

        # 按得分选择前 k% 的节点
        num_nodes = x.size(0)
        k = max(int(num_nodes * self.pool_ratio), 1)  # 确保至少保留一个节点
        topk_indices = scores.topk(k, sorted=False).indices  # Top-k 节点索引

        # 筛选重要节点和边
        x_pooled = x[topk_indices]
        region_indices_pooled = region_indices[topk_indices]  # 更新 region_indices

        mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        mask[topk_indices] = True
        edge_index, edge_attr = self.filter_edges(edge_index, edge_attr, mask)

        # 更新 batch
        batch_pooled = batch[topk_indices]

        return x_pooled, edge_index, edge_attr, region_indices_pooled, batch_pooled, scores

    def filter_edges(self, edge_index, edge_attr, mask):
        # 根据节点掩码筛选边
        edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
        return edge_index[:, edge_mask], edge_attr[edge_mask]

# 定义 BrainGNNBlock
class BrainGNNBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
                 num_regions: int, num_communities: int,
                 activation_cls: nn.Module= nn.ReLU, dropout: float=0.1, pool_ratio: float=0.5):
        super(BrainGNNBlock, self).__init__()
        self.conv = RaGConv(input_dim, output_dim, num_regions, num_communities)
        self.nonlinear = activation_cls()
        self.bn = nn.BatchNorm1d(output_dim)  # 使用标准的 BatchNorm1d
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if pool_ratio < 0.99:
            self.pool = RPool(output_dim, pool_ratio=pool_ratio)
        else:
            self.pool = None

    def forward(self, graphs: Batch):
        # 需要确保 region_indices 包含在 graphs 中
        if not hasattr(graphs, 'region_indices'):
            raise ValueError("Input Batch must contain 'region_indices'")
        
        x = graphs.x
        edge_index = graphs.edge_index
        edge_attr = graphs.edge_attr
        region_indices = graphs.region_indices
        batch = graphs.batch

        # 卷积操作
        x = self.conv(x, edge_index, edge_attr, region_indices)
        x = self.nonlinear(x)
        x = self.bn(x)
        x = self.dropout(x)

        # 池化操作
        if self.pool is not None:
            x, edge_index, edge_attr, region_indices, batch, _ = self.pool(x, edge_index, edge_attr, region_indices, batch)
            graphs = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            graphs.region_indices = region_indices
        else:
            graphs = Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            graphs.region_indices = region_indices

        return graphs

# 定义 BrainGNN
class BrainGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_regions, num_classes, num_communities=8, pool_ratio=0.5):
        super(BrainGNN, self).__init__()
        layers_dict = OrderedDict()
        self.num_nodes_initial = solve_num_nodes(num_features)
        self.output_dim = hidden_dim  # 最终输出维度
        last_hidden_dim = num_features
        last_num_nodes = self.num_nodes_initial

        # 假设 hidden_dims 定义了每一层的隐藏维度
        hidden_dims = [hidden_dim]  # 这里简化为单层
        for i, hidden_dim in enumerate(hidden_dims):
            layers_dict[f'brain_gnn_block_{i}'] = BrainGNNBlock(
                input_dim=last_hidden_dim, 
                output_dim=hidden_dim, 
                num_regions=num_regions,
                num_communities=num_communities,
                dropout=0.1,
                pool_ratio=pool_ratio
            )
            last_hidden_dim = hidden_dim
            # last_num_nodes = ...  # 更新节点数，如果需要

        self.gnn_layers = nn.Sequential(layers_dict)
        self.pool = lambda x, batch: torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data: torch.Tensor, region_indices: torch.Tensor):
        """
        Args:
            data (torch.Tensor): (num_nodes, input_dim)
            region_indices (torch.Tensor): (num_nodes, ) 每个节点对应的区域索引
        Returns:
            torch.Tensor: 分类结果
        """
        # 将张量转换为 Batch 对象
        graphs = tensor_to_batch_graph(data)  # 假设此函数返回包含 x, edge_index, edge_attr, batch 的 Batch 对象
        graphs.region_indices = region_indices  # 将 region_indices 添加到 Batch 对象

        # 通过 GNN 层
        graphs = self.gnn_layers(graphs)

        # 特征汇总
        features = self.pool(graphs.x, graphs.batch)  # (batch_size, hidden_dim * 2)

        # 分类
        out = self.mlp(features)
        return out