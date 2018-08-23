#This code written by amaotone
#https://amalog.hateblo.jp/entry/kaggle-feature-management

import re
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
import numpy as np

import argparse
import inspect

def get_arguments(description = None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f'./data/{self.name}_train.ftr'
        self.test_path = Path(self.dir) / f'./data/{self.name}_test.ftr'
    
    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self
    


    @abstractmethod
    def create_features(self):
        raise NotImplementedError
    
    def save(self):
        try:
            self.train.to_feather(str(self.train_path))
            self.test.to_feather(str(self.test_path))
        except:
            self.train.reset_index(inplace=True)
            self.test.reset_index(inplace=True)
            self.train.to_feather(str(self.train_path))
            self.test.to_feather(str(self.test_path))

def get_input(feats=None,converting=False,debug=False):
    if converting:
        dfs = [pd.read_feather(f'./data/{f}_train.ftr') for f in feats]
        X_train = pd.concat(dfs, axis=1)

        dfs = [pd.read_feather(f'./data/{f}_test.ftr') for f in feats]
        X_test = pd.concat(dfs, axis=1)

        if debug:
            X_train = X_train.loc[:5000]
            X_test = X_test.loc[:1000]
    else:
        X_train = pd.read_csv('./input/application_train.csv')
        X_test = pd.read_csv('./input/application_test.csv')  
        X_train = clean_data(X_train)
        X_test = clean_data(X_test)

    return X_train,X_test   

def clean_data(df):
    df = df[df["CODE_GENDER"] != 'XNA']
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    return df

def get_flagdoc_columns():
    return ['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']

