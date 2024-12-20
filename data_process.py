from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from config import PipelineConfig

TARGETS_FILE = 'prediction_variables.txt'
SUBJECT_IDS_FILE = 'HCP_list_Yeo.txt'
BEHAVIOR_DATA_FILE = 'HCP_s1200.csv'
DATA_FILE = 'rsfc_atlas400_753_4.npy'
SUBJECT_KEY = '30Subject'

class HCPDataset(Dataset):
    def __init__(self, config: PipelineConfig):
        self.root = config.data_dir
        self.targets = config.pred_vars
        
        self.target2id = {target.strip(): i for i, target in enumerate(self.targets)}
        self.id2target = {i: target.strip() for i, target in enumerate(self.targets)}
        
        with open(self.root / SUBJECT_IDS_FILE) as f:
            self.subjects = [int(s) for s in f.read().splitlines()]
        self.subject2id = {s: id_ for id_, s in enumerate(self.subjects)}
        self.id2subject = {id_: s for id_, s in enumerate(self.subjects)}
        
        print(f"Loading behavioral data from {BEHAVIOR_DATA_FILE}")
        self.behav_df = pd.read_csv(self.root / BEHAVIOR_DATA_FILE)
        self.behav_df = self.behav_df[self.targets + [SUBJECT_KEY]]
        self.behav_df = self.behav_df.sort_values(by=self.targets).reset_index(drop=True)
        print(f"Done! Shape: {self.behav_df.shape}")
        
        print(f"Loading brain data from {DATA_FILE}")
        self.brain_data = np.load(self.root / DATA_FILE)
        print(f"Done! Shape: {self.brain_data.shape}")
        
        config.input_dim = self.dim_brain
        config.backbone_config.input_dim = self.dim_brain
        config.backbone_config.target_dim = self.dim_behav
        
    @property
    def dim_behav(self):
        return len(self.targets)
    
    @property
    def dim_brain(self):
        return self.brain_data.shape[1]
    
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, index: int|slice):
        if isinstance(index, slice):
            return [self[i] for i in range(index.start, index.stop, index.step)]
        bahav_data = self.behav_df[self.behav_df[SUBJECT_KEY] == self.id2subject[index]]
        bahav_data = np.array(bahav_data[self.targets].values)  # (1, num_targets)
        bahav_data = bahav_data.squeeze() # (num_targets,)
        func_data = self.brain_data[index]  # (dim_features,)
        return {
            'input': func_data,
            'labels': bahav_data  
        }
