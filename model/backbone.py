import torch
import torch.nn as nn
import pandas as pd
from config import *
from collections import OrderedDict

def get_backbone(config: PipelineConfig):
    match config.backbone_config.backbone_type:
        case 'fnn':
            return FNN(config.backbone_config)
        case _:
            raise ValueError(f"Backbone type {config.backbone_config.backbone_type} not supported")

class FCBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
                 activation_cls: nn.Module= nn.ReLU, dropout: float=0.5):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = activation_cls()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, data: torch.Tensor):
        data = self.fc(data)
        data = self.relu(data)
        data = self.dropout(data)
        return data

class FNN(nn.Module):
    def __init__(self, config: FNNConfig):
        super(FNN, self).__init__()
        layers_dict = OrderedDict()
        last_hidden_dim = config.input_dim
        for i, hidden_dim in enumerate(config.hidden_dims):
            layers_dict[f'fcblock_{i}'] = FCBlock(last_hidden_dim, hidden_dim)
            last_hidden_dim = hidden_dim
        self.fc_layers = nn.Sequential(layers_dict)
        
    def forward(self, data: torch.Tensor, attention_mask: torch.Tensor=None):
        """_summary_

        Args:
            data (torch.Tensor): (batch_size, input_dim)
            attention_mask (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        features = self.fc_layers(data)
        return features
