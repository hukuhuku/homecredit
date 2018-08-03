import pandas as pd
import numpy as np

from base import *
from base import Feature,Data

class OneHotEncoding(Feature):
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
        tmp_train = pd.DataFrame()
        tmp_test = pd.DataFrame()
        tmp_train, cat_cols = self.one_hot_encoder(train, nan_as_category=False)
        tmp_test,cat_cols = self.one_hot_encoder(test,nan_as_category= False)
        tmp_train.to_csv("./data/sada.csv",index=True,header=True)
        self.train = tmp_train.reset_index()
        self.test = tmp_test.reset_index()
    

if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_csv('./input/application_train.csv')
    test = pd.read_csv('./input/application_test.csv')
    train = train[train["CODE_GENDER"] != 'XNA']
    test = test[test["CODE_GENDER"] != 'XNA']
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        train[bin_feature], uniques = pd.factorize(train[bin_feature])
        test[bin_feature], uniques = pd.factorize(test[bin_feature])
    generate_features(globals(), args.force)


