import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent/ 'data' / 'rsfc_atlas400_753_4.txt'

data = np.loadtxt(DATA_PATH)
np.save(DATA_PATH.parent / (DATA_PATH.stem + '.npy'), data)