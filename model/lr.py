import torch
import torch.nn as nn
from config import *
from collections import OrderedDict
from .mlp import FCBlock

def get_kernel(kernel_type: str):
    match kernel_type:
        case 'rbf':
            return lambda x, y: torch.exp(-torch.norm(x - y, dim=-1) ** 2)
        case 'linear':
            return lambda x, y: torch.sum(x * y, dim=-1)
        case 'polynomial':
            return lambda x, y, p=3: (torch.sum(x * y, dim=-1) + 1) ** p
        case 'sigmoid':
            return lambda x, y, alpha=1, c=0: torch.tanh(alpha * torch.sum(x * y, dim=-1) + c)
        case _:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

class LinearRegression(nn.Module):
    def __init__(self, config: LinearRegressionConfig):
        super(LinearRegression, self).__init__()
        
    def forward(self, data: torch.Tensor):
        return data
    
class PartialLinearRegression(nn.Module):
    def __init__(self, config: LinearRegressionConfig):
        super(PartialLinearRegression, self).__init__()
        nonlinear_part = OrderedDict()
        last_hidden_dim = config.input_dim
        for i, hidden_dim in enumerate(config.hidden_dims):
            nonlinear_part[f'fcblock_{i}'] = FCBlock(last_hidden_dim, hidden_dim, dropout=config.dropout)
            last_hidden_dim = hidden_dim
        linear_part = nn.Linear(config.input_dim, last_hidden_dim, bias=False)
        self.nonlinear = nn.Sequential(nonlinear_part)
        self.linear = linear_part
        
        
    def forward(self, data: torch.Tensor):
        return self.linear(data) + self.nonlinear(data)