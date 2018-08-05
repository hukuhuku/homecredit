import pandas as pd
import numpy as np

from base import *
from base import Feature
import gc; 

def one_hot_encoder(df, nan_as_category = True):
    """
    データフレームの内，型がobjectの列をone hot 化.
    引数
    df: データフレーム
    出力
    df: カテゴリ変数をone hot 化ｓたデータフレーム
    new_columns: 新しく作成した列の名前
    """
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df[new_columns], new_columns

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
        tmp_df, cat_cols = one_hot_encoder(df, nan_as_category=False)
    
        tmp_df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'] # 勤続日数/年齢日数
        tmp_df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT'] # 総収入/借入額
        tmp_df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'] # 総収入/家族人数
        tmp_df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'] # 月々の返済額/総収入
        tmp_df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'] # 月々の返済額/借入額

        self.df = tmp_df.reset_index()
   
        del (tmp_df);gc.collect()

class bureau_and_balance(Feature):
    def create_features(self,nan_as_category =True):
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


if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_csv('./input/application_train.csv')
    test = pd.read_csv('./input/application_test.csv')
    df = concat_data(train,test)

    bureau = pd.read_csv("./input/bureau.csv")
    bb = pd.read_csv('./input/bureau_balance.csv')

    generate_features(globals(), args.force)


