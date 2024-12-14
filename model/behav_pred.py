import torch
import torch.nn as nn
import torch.nn.parallel
from .backbone import get_backbone
from data_process import HCPDataset

class BehaviorPredictionModel(nn.Module):
    def __init__(self, config):
        super(BehaviorPredictionModel, self).__init__()
        self.backbone = get_backbone(config)
        self.predictor = nn.LazyLinear(config.target_dim)
        self._lazy_init()
        self.backbone = torch.nn.DataParallel(self.backbone) 
        self.predictor = torch.nn.DataParallel(self.predictor)  
        
    def _lazy_init(self):
        input_dim = HCPDataset.func_dim()
        dummy_input = torch.randn(1, input_dim)
        self.forward(dummy_input)
        
    def forward(self, 
                input: torch.Tensor, 
                labels: torch.Tensor=None,):
        functional_data = input.float()
        features = self.backbone(functional_data)
        predictions = self.predictor(features)
        loss = None
        if labels is not None:
            labels = labels.float().squeeze()
            loss = nn.MSELoss()(predictions, labels)
        return {'loss': loss, 'logits': predictions}