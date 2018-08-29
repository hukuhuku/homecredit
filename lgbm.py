import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.model_selection import GridSearchCV

from base import *
import tqdm

def get_clf():
    # LightGBM parameters found by Bayesian optimization
    return LGBMClassifier(
            n_jobs = -1,
            #is_unbalance=True,
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
    
def get_drop_columns(train_df,test_df):
    # Divide in training/validation and test data
    print("Starting LightGBM. Train shape: {}".format(train_df.shape,))

    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    sub = test_df[["SK_ID_CURR"]]
    target = train_df["TARGET"]
    train_df = train_df.drop(['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index'],axis=1)
    test_df = test_df.drop(['SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index'],axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(train_df, target, random_state=0)
    print(train_x.shape)
    clf = get_clf()

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
        eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

    oof_preds = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

    print(len(clf.feature_importances_))
    print(len(feats))
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = clf.feature_importances_
  
    gc.collect()
    score = roc_auc_score(valid_y, oof_preds)
    print('Full AUC score %.6f' %score)

    drop_columns = fold_importance_df.loc[fold_importance_df["importance"] ==0 ,"feature"]

    sub["TARGET"] = clf.predict_proba(test_df, num_iteration=clf.best_iteration_)[:, 1]
    sub[["SK_ID_CURR","TARGET"]].to_csv("./output/lgbm1.csv", index= False)

    del clf, train_x, train_y, valid_x, valid_y;gc.collect()
    return drop_columns


def fit_predict(train,test,num_folds,debug):
    models = []
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train.shape, test.shape))
    gc.collect()
    
    folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train.shape[0])
    sub_preds = np.zeros(test.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    sub = test[["SK_ID_CURR"]]
    target = train["TARGET"]
    train = train.drop(['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index'],axis=1)
    test = test.drop(['SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index'],axis=1)

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[feats],target)):
        train_x, train_y = train.iloc[train_idx], target.iloc[train_idx]
        valid_x, valid_y = train.iloc[valid_idx], target.iloc[valid_idx]

        print(train_x.shape)
        clf = get_clf()

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

    
    score = roc_auc_score(target, oof_preds)
    print('Full AUC score {}'.format(score))
    # Write submission file and plot feature importance

    #if not debug:
    sub['TARGET'] = sub_preds
    sub[['SK_ID_CURR', 'TARGET']].to_csv("./output/lgbm.csv", index= False)

    return 

  
def main(debug):
    
    feats = ["application","bureau_and_balance","pos_cash",
            "installments_payments","credit_card_valance",
            "previous_aplications","corresponding_aggregate"]

    name = "_and_".join(feats)
    print(name+" modeling start")

    with timer("Load_data"):
        train,test = get_input(feats,converting=True,debug=debug)

    with timer("Get Drop Columns"):
        drop_columns = get_drop_columns(train,test)
        train = train.drop(drop_columns,axis=1)
        test = test.drop(drop_columns,axis=1)
    
    with timer("Run LightGBM with kfold"):
        fit_predict(train,test,debug=debug,num_folds=3)

    
    
if __name__ == "__main__":
    main(debug=False)
