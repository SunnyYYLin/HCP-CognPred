from pathlib import Path
from dataclasses import dataclass, field

DATA_DIR = Path(__file__).parent / 'data'

@dataclass
class FNNConfig:
    backbone_type: str = 'fnn'
    hidden_dims: list[int] = field(default_factory=lambda: [512, 128, 32])
    dropout: float = 0.1

BackboneConfig = FNNConfig

@dataclass
class PipelineConfig:
    data_dir: str = DATA_DIR
    input_dim: int = -1
    target_dim: int = -1
    pred_vars: list[str] = field(default_factory=lambda: [])
    backbone_config: BackboneConfig = None
    
    def __post_init__(self):
        self.target_dim = len(self.pred_vars)