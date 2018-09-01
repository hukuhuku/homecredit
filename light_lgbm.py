import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from base import *
import tqdm

def lightgbm(train_df):
    # Divide in training/validation and test data
    print("Starting LightGBM. Train shape: {}".format(train_df.shape,))

    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    train_x, valid_x, train_y, valid_y = train_test_split(train_df[feats], train_df["TARGET"], random_state=0)

    clf = LGBMClassifier(
        n_jobs=-1,
        n_estimators=10000,
        learning_rate=0.02,
        num_leaves=34,
        colsample_bytree=0.9497036,
        subsample=0.8715623,
        max_depth=8,
        reg_alpha=0.041545473,
        reg_lambda=0.0735294,
        min_split_gain=0.0222415,
        min_child_weight=39.3259775,
        silent=-1,
        verbose=-1, 
    )

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
        eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

    oof_preds = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = clf.feature_importances_
    
  
    gc.collect()
    score = roc_auc_score(valid_y, oof_preds)
    print('Full AUC score %.6f' %score)

    fold_importance_df.sort_values("importance",ascending=False).to_csv("./importance/importance_{}.csv".format(score))

    del clf, train_x, train_y, valid_x, valid_y;gc.collect()


def main():
    feats = ["application","bureau_and_balance","pos_cash",
            "installments_payments","credit_card_valance",
            "external_score_statics","previous_aplications",
            "income","application_and_bureau"]

    name = "_and_".join(feats)
    print(name+" modeling start")

    with timer("Load_data"):
        train,_ = get_input(feats,converting=True)
        train = train.sample(n=(int(train.shape[0]/10)))
    print(train.shape)

    with timer("Run LightGBM"):
        importance = lightgbm(train)

    
    
if __name__ == "__main__":
    main()