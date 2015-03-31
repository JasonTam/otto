import numpy as np
import os

CUR_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(CUR_DIR, '../data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')


def load_csv(f_path=TRAIN_PATH):
    raw = np.loadtxt(f_path, dtype='str', delimiter=',', skiprows=1)
    ids = raw[:, 0].astype('int')
    feats = raw[:, 1:94].astype('int')
    if raw.shape[1] == 95:
        labels = np.array([l.split('_')[1] 
                           for l in raw[:, -1]]).astype('int')
        return ids, feats, labels
    else:
        return ids, feats
