import pandas as pd
import numpy as np

from base import *
from base import Feature
import gc; 

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def concat_data(df,test):
    df = df.append(test).reset_index()
    del(test)
    df = df[df["CODE_GENDER"] != 'XNA']
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    return df

class application(Feature):
    def create_features(self):
        train = pd.read_csv('./input/application_train.csv')
        test = pd.read_csv('./input/application_test.csv')
        df = concat_data(train,test)
        df, cat_cols = one_hot_encoder(df, nan_as_category=False)
    
        df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'] # 勤続日数/年齢日数
        df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT'] # 総収入/借入額
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'] # 総収入/家族人数
        df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'] # 月々の返済額/総収入
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'] # 月々の返済額/借入額

        self.df = df.reset_index()
   
        del (df);gc.collect()

class bureau_and_balance(Feature):
    def create_features(self,nan_as_category =True):
        bureau = pd.read_csv("./input/bureau.csv")
        bb = pd.read_csv('./input/bureau_balance.csv')
        
        bureau,bureau_cat = one_hot_encoder(bureau,nan_as_category)
        bb,bb_cat = one_hot_encoder(bb,nan_as_category)
 
        bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
        for col in bb_cat:
            bb_aggregations[col] = ["mean"]
  
        bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_aggregations)
        bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
        bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
        bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
        del bb, bb_agg
        gc.collect()

        # Bureau and bureau_balance numeric features
        num_aggregations = {
            'DAYS_CREDIT': ['mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM': ['mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_ANNUITY': ['max', 'mean'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'MONTHS_BALANCE_MIN': ['min'],
            'MONTHS_BALANCE_MAX': ['max'],
            'MONTHS_BALANCE_SIZE': ['mean', 'sum']
        }

        # Bureau and bureau_balance categorical features
        cat_aggregations = {}
        for cat in bureau_cat: cat_aggregations[cat] = ['mean']
        for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
        bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
        # Bureau: Active credits - using only numerical aggregations
        active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
        active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
        active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
        active_agg.reset_index(inplace=True)
        bureau_agg.reset_index(inplace=True)
        bureau_agg = pd.merge(bureau_agg,active_agg, how='left', on='SK_ID_CURR')
        del active, active_agg
        gc.collect()
        # Bureau: Closed credits - using only numerical aggregations
        closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
        closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
        closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
        closed_agg.reset_index(inplace=True)
        bureau_agg = pd.merge(bureau_agg,closed_agg, how='left', on='SK_ID_CURR')
        del closed, closed_agg, bureau
        gc.collect()
        bureau_agg.to_csv("tes.csv")
        self.df = bureau_agg


if __name__ == '__main__':
    args = get_arguments()

    


    generate_features(globals(), args.force)


