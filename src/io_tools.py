import numpy as np
import os
import cPickle as pickle
import gzip
from lasagne import layers
import time


CUR_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(CUR_DIR, '../data')
SUBS_DIR = os.path.join(CUR_DIR, '../submissions')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
SAVE_PATH = '/media/raid_arr/data/otto/saved'


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
        

def save_nn(l, f_name, f_dir=SAVE_PATH):
    params = layers.get_all_param_values(l)
    path = os.path.join(f_dir, f_name)
    pickle.dump(params, open(path, 'wb'))
    return path
    

def load_nn_params(l, f_name, f_dir=SAVE_PATH):
    params = pickle.load(open(os.path.join(f_dir, f_name), 'wb'))
    return params


def make_submission(pred, f_name=str(int(time.time()))+'.csv', f_dir=SUBS_DIR):
    save_path = os.path.join(f_dir, f_name)
    header = ','.join(['id'] + ['Class_' + str(ii) for ii in range(1, 10)])

    save_arr = np.c_[np.arange(len(pred), dtype=int)+1, pred]
    fmt = ['%d'] + ['%g']*9
    np.savetxt(save_path, save_arr, delimiter=",", fmt=fmt, 
        header=header, comments='')
    return save_path





