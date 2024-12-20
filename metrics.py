import torch
from torchmetrics import MeanAbsolutePercentageError, PearsonCorrCoef
from transformers import EvalPrediction
from config import PipelineConfig

class CognPredMetrics:
    def __init__(self, config: PipelineConfig):
        self.pred_vars = config.pred_vars
        self.mape = MeanAbsolutePercentageError()
        self.pearson = PearsonCorrCoef(num_outputs=config.target_dim)
        
    def __call__(self, preds: EvalPrediction):
        """
        Args:
            preds (EvalPrediction): preds.labels, preds.predictions
        """
        labels = torch.tensor(preds.label_ids, dtype=torch.float32)
        predictions = torch.tensor(preds.predictions, dtype=torch.float32)
        results = {}
        
        # Calculate MAPE
        results['mape'] = self.mape(predictions, labels)
        for i, target in enumerate(self.pred_vars):
            results[f"{target}_mape"] = self.mape(predictions[:, i], labels[:, i])
        # Calculate Pearson
        pearson = self.pearson(predictions, labels)
        for i, target in enumerate(self.pred_vars):
            results[f"{target}_pearson"] = pearson[i]
        
        return results
        