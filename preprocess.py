import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

def to_npy():
    DATA_PATH = Path(__file__).parent/ 'data' / 'rsfc_Yeo400_753_GSR.txt'

    data = np.loadtxt(DATA_PATH)
    np.save(DATA_PATH.parent / (DATA_PATH.stem + '.npy'), data)
    
def pca():
    DATA_PATH = Path(__file__).parent/ 'data' / 'rsfc_Yeo400_753_GSR.npy'
    data = np.load(DATA_PATH)
    pca = PCA(n_components=256)
    pca.fit(data)
    np.save(DATA_PATH.parent / (DATA_PATH.stem + '_pca.npy'), pca.transform(data))
    
if __name__ == '__main__':
    pca()