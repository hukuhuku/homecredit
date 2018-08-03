import pandas as pd
import numpy as np

from base import *
from base import Feature,Data
import gc; 

def prepare_data(train,test):
    train = train[train["CODE_GENDER"] != 'XNA']
    test = test[test["CODE_GENDER"] != 'XNA']
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        train[bin_feature], uniques = pd.factorize(train[bin_feature])
        test[bin_feature], uniques = pd.factorize(test[bin_feature])
    train['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    return train,test

class application(Feature):
    def one_hot_encoder(self,df, nan_as_category = True):
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

    def create_features(self):
        tmp_train, cat_cols = self.one_hot_encoder(train, nan_as_category=False)
        tmp_test,cat_cols = self.one_hot_encoder(test,nan_as_category= False)
        
        tmp_train['DAYS_EMPLOYED_PERC'] = train['DAYS_EMPLOYED'] / train['DAYS_BIRTH'] # 勤続日数/年齢日数
        tmp_train['INCOME_CREDIT_PERC'] = train['AMT_INCOME_TOTAL'] / train['AMT_CREDIT'] # 総収入/借入額
        tmp_train['INCOME_PER_PERSON'] = train['AMT_INCOME_TOTAL'] / train['CNT_FAM_MEMBERS'] # 総収入/家族人数
        tmp_train['ANNUITY_INCOME_PERC'] = train['AMT_ANNUITY'] / train['AMT_INCOME_TOTAL'] # 月々の返済額/総収入
        tmp_train['PAYMENT_RATE'] = train['AMT_ANNUITY'] / train['AMT_CREDIT'] # 月々の返済額/借入額

        tmp_test['DAYS_EMPLOYED_PERC'] = test['DAYS_EMPLOYED'] / test['DAYS_BIRTH'] # 勤続日数/年齢日数
        tmp_test['INCOME_CREDIT_PERC'] = test['AMT_INCOME_TOTAL'] / test['AMT_CREDIT'] # 総収入/借入額
        tmp_test['INCOME_PER_PERSON'] = test['AMT_INCOME_TOTAL'] / test['CNT_FAM_MEMBERS'] # 総収入/家族人数
        tmp_test['ANNUITY_INCOME_PERC'] = test['AMT_ANNUITY'] / test['AMT_INCOME_TOTAL'] # 月々の返済額/総収入
        tmp_test['PAYMENT_RATE'] = test['AMT_ANNUITY'] / test['AMT_CREDIT'] # 月々の返済額/借入額
        self.train = tmp_train.reset_index()
        self.test = tmp_test.reset_index()
        del (tmp_train,tmp_test);gc.collect()

if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_csv('./input/application_train.csv')
    test = pd.read_csv('./input/application_test.csv')
    train,test = prepare_data(train,test) 

    generate_features(globals(), args.force)


