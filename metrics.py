import torch
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef
from transformers import EvalPrediction
from config import PipelineConfig

class CognPredMetrics:
    def __init__(self, config: PipelineConfig):
        self.pred_vars = config.pred_vars
        self.mae = MeanAbsoluteError()
        self.pearson = PearsonCorrCoef(num_outputs=config.target_dim)
        
    def __call__(self, preds: EvalPrediction):
        """
        Args:
            preds (EvalPrediction): preds.labels, preds.predictions
        """
        labels = torch.tensor(preds.label_ids, dtype=torch.float32)
        predictions = torch.tensor(preds.predictions, dtype=torch.float32)
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)
        
        results = {}
        
        # Calculate MAPE
        results['mae'] = self.mae(predictions, labels)
        for i, target in enumerate(self.pred_vars):
            results[f"{target}_mae"] = self.mae(predictions[:, i], labels[:, i])
            
        # Calculate Pearson
        pearson = self.pearson(predictions, labels)
        results['pearson'] = pearson.mean().item()
        for i, target in enumerate(self.pred_vars):
            results[f"{target}_pearson"] = pearson[i] \
                if pearson.shape[0] > 1 else pearson.item()
        
        return results
        