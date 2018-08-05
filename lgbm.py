import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold

import tqdm


def load_data(feats):
    f = feats.pop()
    df = pd.read_feather(f'./data/{f}_df.ftr')
    for f in feats:
        df = pd.merge(df,pd.read_feather(f'./data/{f}_df.ftr'),how="left",on="SK_ID_CURR")
    return df

def main():
    feats = ["application","bureau_and_balance"]
    df = load_data(feats)

if __name__ == "__main__":

    main()
