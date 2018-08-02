import pandas as pd
import numpy as np

from base import *
from base import Data



class avg_bureau_balance(Data):
    def create_data(self):
        tmp_buro_bal = pd.concat([buro_bal, pd.get_dummies(buro_bal.STATUS, prefix='buro_bal_status')], axis=1).drop('STATUS', axis=1)
        buro_counts = tmp_buro_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
        tmp_buro_bal['buro_count'] = tmp_buro_bal['SK_ID_BUREAU'].map(buro_counts['MONTHS_BALANCE'])
        avg_buro_bal = tmp_buro_bal.groupby('SK_ID_BUREAU').mean()
        avg_buro_bal.columns = ['avg_buro_' + f_ for f_ in avg_buro_bal.columns]
        self.data = avg_buro_bal

if __name__ == '__main__':
    args = get_arguments()

    buro_bal = pd.read_csv('./input/bureau_balance.csv')


    generate_data(globals(), args.force)