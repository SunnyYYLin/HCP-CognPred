from pathlib import Path
from dataclasses import dataclass, field

DATA_DIR = Path(__file__).parent / 'data'

@dataclass
class FNNConfig:
    backbone_type: str = 'fnn'
    hidden_dims: list[int] = field(default_factory=lambda: [512, 128, 32])
    dropout: float = 0.1
    
    def abbrev(self):
        return f"fnn_hidden[{','.join(map(str, self.hidden_dims))}]_dropout{self.dropout}"
    
    @classmethod
    def from_abbrev(cls, abbrev: str):
        params = abbrev.split('_')
        hidden_dims = params[1].removeprefix('hidden')
        hidden_dims = hidden_dims[1:-1].split(',')
        hidden_dims = list(map(int, hidden_dims))
        dropout = float(params[2].removeprefix('dropout'))
        return cls(hidden_dims=hidden_dims, dropout=dropout)

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
        
    def abbrev(self):
        return f"[{','.join(self.pred_vars)}]_by_{self.backbone_config.abbrev()}"
    
    @classmethod
    def from_abbrev(cls, abbrev: str):
        pred_vars, _, backbone = abbrev.partition('_by_')
        pred_vars = pred_vars[1:-1].split(',')
        backbone = BackboneConfig.from_abbrev(backbone)
        return cls(pred_vars=pred_vars, backbone_config=backbone)