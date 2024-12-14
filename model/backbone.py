import torch
import torch.nn as nn
import pandas as pd
from config import *
from collections import OrderedDict

def get_backbone(config: BackboneConfig):
    if config.backbone_type == 'cnn':
        return FeatureExtractor_conv(config)
    elif config.backbone_type == 'fc':
        return FeatureExtractor_fnn(config)
    else:
        raise ValueError(f"Unsupported backbone type: {config.backbone_type}")

class ConvBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvBlock1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        return x

class FeatureExtractor_conv(nn.Module):
    def __init__(self, config: CNNConfig):
        super(FeatureExtractor_conv, self).__init__()
        conv_blocks = OrderedDict(
            [
                (
                    f'conv_{i}', 
                    ConvBlock1d(
                        1 if i==0 else config.num_filters[i-1], 
                        config.num_filters[i], 
                        config.kernel_sizes[i]
                    )
                )
                for i in range(len(config.kernel_sizes))
            ]
        )
        self.convs = nn.Sequential(conv_blocks)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(config.num_filters[-1], config.target_dim)
        
    def forward(self, data: torch.Tensor, attention_mask: torch.Tensor=None):
        """_summary_

        Args:
            data (torch.Tensor): (batch_size, seq_len)
            attention_mask (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        #data = data.float()
        features = data.unsqueeze(1)
        features = self.convs(features) # (batch_size, num_filters[-1], seq_len)
        features = self.pool(features).squeeze(-1) # (batch_size, num_filters[-1])
        features = self.dropout(features)
        features = self.fc(features) # (batch_size, target_dim)
        return features

class FeatureExtractor_fnn(nn.Module):
    def __init__(self, config: FNNConfig):
        super(FeatureExtractor_fnn, self).__init__()
        layers = []
        input_dim = config.input_dim
        for output_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, config.target_dim))
        self.fc_layers = nn.Sequential(*layers)
        
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
