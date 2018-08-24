#Almost https://www.kaggle.com/dromosys/fork-of-fork-lightgbm-with-simple-features-cee847/code

import pandas as pd
import numpy as np

from base import *
import gc

import warnings
warnings.filterwarnings('ignore')

train,test = get_input(converting=False)

train_id = train[['SK_ID_CURR']]
test_id = test[['SK_ID_CURR']]

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df,new_columns

class external_score_statics(Feature):
    def function(self,df):
        tmp = pd.DataFrame()
        tmp['SOURCES_PROD_PRODUCT'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        tmp['EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        tmp['SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
        tmp['SCORES_STD'] = tmp['SCORES_STD'].fillna(tmp['SCORES_STD'].mean())
        return tmp
    def create_features(self):
        self.train = self.function(train)
        self.test = self.function(test)


class application(Feature):
    def function(self,df):
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

        docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
        live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    
        df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'] # 勤続日数/年齢日数
        df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT'] # 総収入/借入額

        df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
        df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
        
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'] # 総収入/家族人数
        df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'] # 月々の返済額/総収入
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'] # 月々の返済額/借入額
        
        df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY'] #何回に分けて返済するか
        df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'] #商品の値段に対するクレジットの割合
        df['INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])#子供に対するクライアントの収入
        
        df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']#車を早く持てる＝＞家庭環境良
        df['CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
        df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
        df['PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']

        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            df[bin_feature], uniques = pd.factorize(df[bin_feature])

        df, cat_cols = one_hot_encoder(df, nan_as_category=True)

        return df

    def create_features(self):
        self.train = self.function(train)
        self.test = self.function(test)
        

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

        self.train = pd.merge(train_id,bureau_agg,on="SK_ID_CURR",how="left").drop(["SK_ID_CURR"],axis=1)
        self.test = pd.merge(test_id,bureau_agg,on="SK_ID_CURR",how="left").drop(["SK_ID_CURR"],axis=1)

        del bureau_agg;gc.collect()

class previous_aplications(Feature):
    def create_features(self,nan_as_category=True):
        prev = pd.read_csv('./input/previous_application.csv')
        prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
        # Days 365.243 values -> nan
        prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
        prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
        prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
        prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
        prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
        # Add feature: value ask / value received percentage
        prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
        # Previous applications numeric features
        num_aggregations = {
            'AMT_ANNUITY': [ 'max', 'mean'],
            'AMT_APPLICATION': [ 'max','mean'],
            'AMT_CREDIT': [ 'max', 'mean'],
            'APP_CREDIT_PERC': [ 'max', 'mean'],
            'AMT_DOWN_PAYMENT': [ 'max', 'mean'],
            'AMT_GOODS_PRICE': [ 'max', 'mean'],
            'HOUR_APPR_PROCESS_START': [ 'max', 'mean'],
            'RATE_DOWN_PAYMENT': ['max', 'mean'],
            'DAYS_DECISION': [ 'max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum'],
        }
        # Previous applications categorical features
        cat_aggregations = {}
        for cat in cat_cols:
            cat_aggregations[cat] = ['mean']
    
        prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

        #Previous Applications: Approved Applications - only numerical features
        approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
        approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
        approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
        approved_agg.reset_index(inplace=True)
        prev_agg.reset_index(inplace=True)
        prev_agg = pd.merge(prev_agg,approved_agg, how='left', on='SK_ID_CURR')

        # Previous Applications: Refused Applications - only numerical features
        refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
        refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
        refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
        refused_agg.reset_index(inplace=True)
        prev_agg = pd.merge(prev_agg,refused_agg, how='left', on='SK_ID_CURR')
        
        del refused, refused_agg, approved, approved_agg, prev
        gc.collect()
        
        self.train = pd.merge(train_id,prev_agg,on="SK_ID_CURR",how="left").drop(["SK_ID_CURR"],axis=1)
        self.test = pd.merge(test_id,prev_agg,on="SK_ID_CURR",how="left").drop(["SK_ID_CURR"],axis=1)


class pos_cash(Feature):
    def create_features(self,nan_as_category=True):
        pos = pd.read_csv('./input/POS_CASH_balance.csv')
        pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
        
        aggregations = {
            'MONTHS_BALANCE': ['max', 'mean', 'size'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']
    
        pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
        pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
        # Count pos cash accounts
        pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
        pos_agg.reset_index(inplace=True)
        del pos;gc.collect()
        
        pos_agg.to_csv("temp.csv")
        self.train = pd.merge(train_id,pos_agg,on="SK_ID_CURR",how="left").drop(["SK_ID_CURR"],axis=1)
        self.test = pd.merge(test_id,pos_agg,on="SK_ID_CURR",how="left").drop(["SK_ID_CURR"],axis=1)


class installments_payments(Feature):
    def create_features(self,nan_as_category=True):
        ins = pd.read_csv('./input/installments_payments.csv')
        ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
        # Percentage and difference paid in each installment (amount paid and installment value)
        ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
        ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
        # Days past due and days before due (no negative values)
        ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
        ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
        ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
        ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
        # Features: Perform aggregations
        aggregations = {
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['max', 'mean', 'sum','min','std' ],
            'DBD': ['max', 'mean', 'sum','min','std'],
            'PAYMENT_PERC': [ 'max','mean',  'var','min','std'],
            'PAYMENT_DIFF': [ 'max','mean', 'var','min','std'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum','min','std'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum','std'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum','std']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']
        ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
        ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
        # Count installments accounts
        ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
        ins_agg.reset_index(inplace=True)
        del ins;gc.collect()
        
        self.train = pd.merge(train_id,ins_agg,on="SK_ID_CURR",how="left").drop(["SK_ID_CURR"],axis=1)
        self.test = pd.merge(test_id,ins_agg,on="SK_ID_CURR",how="left").drop(["SK_ID_CURR"],axis=1)



class credit_card_valance(Feature):
    def create_features(self,nan_as_category = True):
        cc = pd.read_csv('./input/credit_card_balance.csv')
        cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
        # General aggregations
        cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
        cc_agg = cc.groupby('SK_ID_CURR').agg([ 'max', 'mean', 'sum', 'var'])
        cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
        # Count credit card lines
        cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
        cc_agg.reset_index(inplace=True)
        del cc;gc.collect()
        
        self.train = pd.merge(train_id,cc_agg,on="SK_ID_CURR",how="left").drop(["SK_ID_CURR"],axis=1)
        self.test = pd.merge(test_id,cc_agg,on="SK_ID_CURR",how="left").drop(["SK_ID_CURR"],axis=1)


if __name__ == '__main__':
    args = get_arguments()
    generate_features(globals(), args.force)
