import torch
import torch.nn as nn
import torch.nn.parallel
from .backbone import get_backbone
from data_process import HCPDataset

class BehaviorPredictionModel(nn.Module):
    def __init__(self, config):
        super(BehaviorPredictionModel, self).__init__()
        self.input_dim = config.input_dim
        self.backbone = get_backbone(config)
        self.predictor = nn.LazyLinear(config.target_dim)
        self.loss = nn.MSELoss()
        self._lazy_init()
        
    def _lazy_init(self):
        dummy_input = torch.randn(1, self.input_dim)
        self.forward(dummy_input)
        
    def forward(self, 
                input: torch.Tensor, 
                labels: torch.Tensor=None,):
        functional_data = input.float() # (batch_size, input_dim)
        features = self.backbone(functional_data) # (batch_size, target_dim)
        predictions = self.predictor(features) # (batch_size, target_dim)
        results = {'predictions': predictions}
        if labels is not None:
            results['loss'] = self.loss(predictions, labels.float())
        return results