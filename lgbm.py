import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold

from base import *
import tqdm

def fit_predict(train,test, num_folds, stratified = False, debug= False):
    models = []
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train.shape, test.shape))
    gc.collect()
    # Cross validation model

    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train.shape[0])
    sub_preds = np.zeros(test.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = 0

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[feats], train['TARGET'])):
        train_x, train_y = train[feats].iloc[train_idx], train['TARGET'].iloc[train_idx]
        valid_x, valid_y = train[feats].iloc[valid_idx], train['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            n_jobs = 4,
            #is_unbalance=True,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=32,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.04,
            reg_lambda=0.073,
            min_split_gain=0.0222415,
            min_child_weight=40,
            silent=-1,
            verbose=-1,
            #scale_pos_weight=11
            )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 1000, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df["importance"] += clf.feature_importances_
    
    fold_importance_df["importance"] /= 5
    print('Full AUC score %.6f' % roc_auc_score(train['TARGET'], oof_preds))
    # Write submission file and plot feature importance

    #if not debug:
    test['TARGET'] = sub_preds
    test[['SK_ID_CURR', 'TARGET']].to_csv("./output/lgbm.csv", index= False)

    return fold_importance_df.sort_values("importance",ascending=False)
def main(debug=False):
    
    feats = ["application","bureau_and_balance","pos_cash","installments_payments","credit_card_valance"]
    name = "_and_".join(feats)
    print(name+" modeling start")

    with timer("Load_data"):
        train,test = get_input(feats,debug=debug)

    with timer("Run LightGBM with kfold"):
        importance = fit_predict(train,test, num_folds= 5, stratified= False,debug=debug)

    importance.to_csv("importance.csv")
if __name__ == "__main__":
    main(debug=True)
