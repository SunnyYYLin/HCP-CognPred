import torch
import torch.nn as nn
import torch.nn.parallel
from .backbone import get_backbone
from data_process import HCPDataset

class RectifiedR2Loss(nn.Module):
    def __init__(self):
        super(RectifiedR2Loss, self).__init__()
        
    def r2(self, preds, labels):
        preds_mean = torch.mean(preds, dim=0)
        ss_tot = torch.sum((preds - preds_mean) ** 2, dim=0)
        ss_res = torch.sum((preds - labels) ** 2, dim=0)
        r2_per_dim = 1 - ss_res / (ss_tot + 1e-10)
        r2_mean = torch.mean(r2_per_dim)
        return r2_mean
    
    def forward(self, preds, labels):
        return torch.log(1 - self.r2(preds, labels))

class BehaviorPredictionModel(nn.Module):
    def __init__(self, config):
        super(BehaviorPredictionModel, self).__init__()
        self.input_dim = config.input_dim
        self.backbone = get_backbone(config)
        self.predictor = nn.LazyLinear(config.target_dim)
        self.mse = nn.MSELoss()
        self.r2loss = RectifiedR2Loss()
        self.loss = lambda x, y: self.mse(x, y) \
            + config.r2loss_weight*self.r2loss(x, y) \
            + config.penalty_weight*torch.norm(self.predictor.weight, p=2)
        self._lazy_init()
        
    def _lazy_init(self):
        dummy_input = torch.randn(1, self.input_dim).exp()
        self.forward(dummy_input)
        
    def forward(self, 
                input: torch.Tensor, 
                labels: torch.Tensor=None,):
        functional_data = input.float() # (batch_size, input_dim)
        features = self.backbone(functional_data) # (batch_size, target_dim)
        predictions = self.predictor(features) # (batch_size, target_dim)
        results = {'predictions': predictions}
        if labels is not None:
            loss = self.loss(predictions, labels.float())
            results['loss'] = loss
        return results