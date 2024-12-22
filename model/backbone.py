from config import *
from .mlp import MLP
from .gcn import GCN
from .gat import GAT

def get_backbone(config: PipelineConfig):
    match config.backbone_config.backbone_type:
        case 'mlp':
            return MLP(config.backbone_config)
        case 'gcn':
            return GCN(config.backbone_config)
        case 'gat':
            return GAT(config.backbone_config)
        case _:
            raise ValueError(f"Backbone type {config.backbone_config.backbone_type} not supported")
