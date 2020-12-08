# !pip install lightgbm

import gc
import time
from utils import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
from lightgbm.sklearn import LGBMRanker
from sklearn.model_selection import StratifiedKFold, KFold
import warnings
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
warnings.filterwarnings('ignore')

print('loading data')

train = pd.read_pickle('data/train.pkl')
test = pd.read_pickle('data/test_B.pkl')

drop_cols = ['label', 'query_id', 'doc_id']
feats = train.columns.drop(drop_cols).values.tolist()

test.sort_values('query_id', inplace=True, ignore_index=True)
group_sub = test.query_id.value_counts().sort_index().values
X_test = test[feats]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
sub = pd.DataFrame({
    'queryid': test.query_id,
    'documentid': test.doc_id,
    'predict_label': 0.0,
})
oof = []

print('start training')
for i, (trn_idx, val_idx) in enumerate(skf.split(train, train.label)):
    print('----------------------{} fold----------------------'.format(i))
    trn_df = train.iloc[trn_idx].sort_values('query_id', ignore_index=True)
    val_df = train.iloc[val_idx].sort_values('query_id', ignore_index=True)
    X_trn, Y_trn = trn_df[feats], trn_df.label.values
    group_trn = trn_df.query_id.value_counts().sort_index().values
    X_val, Y_val = val_df[feats], val_df.label.values
    group_val = val_df.query_id.value_counts().sort_index().values
    
    ranker = LGBMRanker(
        num_leaves=255,
        learning_rate=0.05,
        n_estimators=100000,
        objective='lambdarank',
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=24,
    )
    
    ranker.fit(
        X_trn, Y_trn,
        group=group_trn,
        eval_metric='ndcg',
        eval_set=[(X_val, Y_val)],
        eval_group=[group_val],
        eval_at=[1, 3, 5, 10],
        early_stopping_rounds=200,
        verbose=500,
    )
    
    sub['predict_label'] += ranker.predict(X_test) / skf.n_splits
    oof.append(pd.DataFrame({
        'query_id': val_df.query_id,
        'doc_id': val_df.doc_id,
        'oof': ranker.predict(X_val),
        'label': val_df.label,
    }))


oof = pd.concat(oof, ignore_index=True)
sub.to_csv('sub_lgb.csv', index=False)
oof.to_pickle('oof_lgb.pkl')

print('finish training')