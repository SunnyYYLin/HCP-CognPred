from dataclasses import dataclass, field
from data_process import HCPDataset

behavioral_dim = HCPDataset.behavioral_dim()
func_dim = HCPDataset.func_dim()
@dataclass
class CNNConfig:
    backbone_type: str = 'cnn'
    kernel_sizes: list[int] = field(default_factory=lambda: [19, 17, 15, 13, 11, 9])
    num_filters: list[int] = field(default_factory=lambda: [32, 64, 128, 256, 512, 1024])
    target_dim: int = behavioral_dim

@dataclass
class FNNConfig:
    backbone_type: str = 'fc'
    input_dim: int = func_dim
    hidden_dims: list[int] = field(default_factory=lambda: [512, 128,32])
    target_dim: int = behavioral_dim

BackboneConfig = CNNConfig | FNNConfig