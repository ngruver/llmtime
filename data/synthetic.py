import os
from glob import glob
import numpy as np
import pandas as pd

def get_synthetic_datasets():
    dss = []
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dss += glob(f'{dir_path}/../datasets/synthetic/*.npy')
    x = []
    labels = []
    for ds in dss:
        s = np.load(ds)[:3]
        x.append(s)
        labels += [ds.split('/')[-1].split('.')[0]+(f'_{i}' if len(s)>1 else '') for i in range(len(s))]
    x = np.concatenate(x)
    # subtract mean
    # x -= np.mean(x, axis=1, keepdims=True)
    data = [pd.Series(x[i],index=pd.RangeIndex(len(x[i]))) for i in range(len(x))]
    synthetic_datasets = {dsname:(dat[:140],dat[140:]) for dsname,dat in zip(labels,data)}
    return synthetic_datasets