from config import *
from .mlp import MLP
from .gnn import GCN

def get_backbone(config: PipelineConfig):
    match config.backbone_config.backbone_type:
        case 'mlp':
            return MLP(config.backbone_config)
        case 'gcn':
            return GCN(config.backbone_config)
        case _:
            raise ValueError(f"Backbone type {config.backbone_config.backbone_type} not supported")
