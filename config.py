from pathlib import Path
from dataclasses import dataclass, field

DATA_DIR = Path(__file__).parent / 'data'

@dataclass
class MLPConfig:
    backbone_type: str = 'mlp'
    hidden_dims: list[int] = field(default_factory=lambda: [512, 128, 32])
    dropout: float = 0.1
    
    def abbrev(self):
        return f"mlp_hidden[{','.join(map(str, self.hidden_dims))}]_dropout{self.dropout}"
    
    @classmethod
    def from_abbrev(cls, abbrev: str):
        params = abbrev.split('_')
        hidden_dims = params[1].removeprefix('hidden')
        hidden_dims = hidden_dims[1:-1].split(',')
        hidden_dims = list(map(int, hidden_dims))
        dropout = float(params[2].removeprefix('dropout'))
        return cls(hidden_dims=hidden_dims, dropout=dropout)
    
@dataclass
class GCNConfig:
    backbone_type: str = 'gcn'
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    num_nodes: list[int] = field(default_factory=lambda: [100, 25])
    dropout: float = 0.1
    
    def abbrev(self):
        return f"gcn_hidden{self.hidden_dims}_nodes{self.num_nodes}_dropout{self.dropout}"
    
    @classmethod
    def from_abbrev(cls, abbrev: str):
        params = abbrev.split('_')
        hidden_dims = eval(params[1].removeprefix('hidden'))
        num_nodes = eval(params[2].removeprefix('nodes'))
        dropout = float(params[3].removeprefix('dropout'))
        return cls(hidden_dims=hidden_dims, num_nodes=num_nodes, dropout=dropout)
    
@dataclass
class GATConfig:
    backbone_type: str = 'gat'
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    num_heads: list[int] = field(default_factory=lambda: [4, 2])
    num_nodes: list[int] = field(default_factory=lambda: [100, 25])
    dropout: float = 0.0
    
    def abbrev(self):
        return f"gat_hidden{self.hidden_dims}_nodes{self.num_nodes}_heads{self.num_heads}_dropout{self.dropout}"
    
    @classmethod
    def from_abbrev(cls, abbrev: str):
        params = abbrev.split('_')
        hidden_dims = eval(params[1].removeprefix('hidden'))
        num_nodes = eval(params[2].removeprefix('nodes'))
        num_heads = eval(params[3].removeprefix('heads'))
        dropout = float(params[4].removeprefix('dropout'))
        return cls(hidden_dims=hidden_dims, num_nodes=num_nodes, num_heads=num_heads, dropout=dropout)

@dataclass
class LinearRegressionConfig:
    backbone_type: str = 'lr'
    
    def abbrev(self):
        return 'lr'
    
    @classmethod
    def from_abbrev(cls, abbrev: str):
        return cls()

@dataclass
class PartialLinearRegressionConfig:
    backbone_type: str = 'plr'
    hidden_dims: list[int] = field(default_factory=lambda: [256, 64])
    dropout: float = 0.0
    
    def abbrev(self):
        return f"plr_hidden[{','.join(map(str, self.hidden_dims))}]_dropout{self.dropout}"
    
    @classmethod
    def from_abbrev(cls, abbrev: str):
        params = abbrev.split('_')
        hidden_dims = params[1].removeprefix('hidden')
        hidden_dims = hidden_dims[1:-1].split(',')
        hidden_dims = list(map(int, hidden_dims))
        dropout = float(params[2].removeprefix('dropout'))
        return cls(hidden_dims=hidden_dims, dropout=dropout)

@dataclass
class KernelRegressionConfig:
    backbone_type: str = 'kernel'
    kernel_type: str = 'rbf'
    
    def abbrev(self):
        return f'kernel_{self.kernel_type}'
    
    @classmethod
    def from_abbrev(cls, abbrev: str):
        return cls()

BackboneConfig = MLPConfig|GCNConfig|GATConfig\
    |LinearRegressionConfig|PartialLinearRegressionConfig|KernelRegressionConfig

@dataclass
class PipelineConfig:
    pca: bool = False
    data_dir: str = DATA_DIR
    input_dim: int = -1
    target_dim: int = -1
    pred_vars: list[str] = field(default_factory=lambda: [])
    r2loss_weight: float = 0.0
    penalty_weight: float = 0.0
    backbone_config: BackboneConfig = None
    
    def __post_init__(self):
        self.target_dim = len(self.pred_vars)
        
    def abbrev(self):
        abbr = f"{self.backbone_config.abbrev()}"
        if self.r2loss_weight > 0:
            abbr += f"_r2w{self.r2loss_weight}"
        if self.penalty_weight > 0:
            abbr += f"_penalty{self.penalty_weight}"
        if self.pca:
            abbr += '_pca'
        return abbr
    
    @classmethod # TODO
    def from_abbrev(cls, abbrev: str):
        backbone, _, pipeline = abbrev.partition('_r2w')
        args = pipeline.split('_')
        r2loss_weight = float(args[0])
        penalty_weight = float(args[1].removeprefix('penalty'))
        pca = 'pca' in args
        backbone = BackboneConfig.from_abbrev(backbone)
        return cls(
            backbone_config=backbone, 
            r2loss_weight=r2loss_weight,
            penalty_weight=penalty_weight,
            pca=pca
        )