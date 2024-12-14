from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / 'data'
TARGETS_FILE = 'target.txt'
SUBJECT_IDS_FILE = 'HCP_list_Yeo.txt'
BEHAVIOR_DATA_FILE = 'HCP_s1200.csv'
DATA_FILES = ['rsfc_atlas400_753_4.txt', 'rsfc_Yeo400_753_GSR.txt', 'scfp_atlas400_753.txt']

class HCPDataset(Dataset):
    def __init__(self, root=DATA_DIR, data_file=DATA_FILES[1]):
        self.root = root
        
        with open(root / TARGETS_FILE) as f:
            self.targets = f.read().splitlines()
        self.target2id = {target.strip(): i for i, target in enumerate(self.targets)}
        self.id2target = {i: target.strip() for i, target in enumerate(self.targets)}
        
        with open(root / SUBJECT_IDS_FILE) as f:
            self.subjects = [int(s) for s in f.read().splitlines()]
        self.subject2id = {s: id_ for id_, s in enumerate(self.subjects)}
        self.id2subject = {id_: s for id_, s in enumerate(self.subjects)}
        
        print(f"Loading behavioral data from {BEHAVIOR_DATA_FILE}")
        self.behavioral_df = pd.read_csv(root / BEHAVIOR_DATA_FILE)
        print(f"Done! Shape: {self.behavioral_df.shape}")
        
        print(f"Loading functional data from {data_file}")
        with open(root / data_file) as f:
            self.functional_data = np.loadtxt(root / data_file)
        print(f"Done! Shape: {self.functional_data.shape}")
        
    @property
    def dim_behav(self):
        return len(self.targets)
    
    @property
    def dim_func(self):
        return self.functional_data.shape[1]
    
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, index):
        bahav_data = self.behavioral_df[self.behavioral_df['Subject'] == self.id2subject[index]]
        bahav_data = np.array(bahav_data[self.targets].values)  # (num_targets,)
        func_data = self.functional_data[index]  # (dim_features,)
        return {
            'input': func_data,
            'labels': bahav_data  }
    @classmethod
    def behavioral_dim(cls, root=DATA_DIR):
        with open(root / TARGETS_FILE) as f:
            targets = f.read().splitlines()
        return len(targets)
    @classmethod
    def func_dim(cls, root=DATA_DIR, data_file=DATA_FILES[1]):
        data_path = root / data_file
        with open(data_path, 'r') as f:
            first_line = f.readline()
        num_columns = len(first_line.strip().split())
        print('fuc_dim')
        print(num_columns)
        return num_columns
