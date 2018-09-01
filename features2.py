import pandas as pd
import numpy as np

from base import *
import gc

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
import lightgbm as lgb

train,test = get_input(converting=False)

train_id = train[['SK_ID_CURR']]
test_id = test[['SK_ID_CURR']]

def target_encode(train,df,target,categorical):
    for col in categorical:
        tmp = pd.DataFrame(train.groupby(col)[target].agg(["mean","sum","count","std","var","max","min","median"])).fillna(-999)
        tmp.reset_index(inplace=True)
        tmp.columns = [col,col+"_mean",col+"_sum",col+"_count",col+"_std",col+"_var",col+"_max",col+"_min",col+"_median"]
        df = pd.merge(df,tmp,how="left",on=col)
        del(tmp)
    return df

class corresponding_aggregate(Feature):
    def function(self,df):
        categorical_cols = [
            "CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY",
            "NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS",
            "OCCUPATION_TYPE"
            ]
        target_cols = [
            "AMT_CREDIT","ANT_ANNUITY","AMT_INCOME_TOTAL"
            "AMT_GOODS_PRICE"
        ]
        dfs = []
        kf = KFold(n_splits = 5,shuffle=True)

        for target in target_cols:
            for categorical in categorical_cols:
                for train_index,test_index in kf.split(df):
                    tmp = target_encode(df.loc[train_index],df.loc[test_index],target,categorical)
                    dfs.append(tmp)

        (pd.concat(dfs,axis=0)).to_csv("tara.csv")
        return pd.concat(dfs,axis=0)
        
    def create_function(self):
        self.train = self.function(,train)



