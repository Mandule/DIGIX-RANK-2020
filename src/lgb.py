import gc
from utils import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import StratifiedKFold
import warnings
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
warnings.filterwarnings('ignore')

print('loading data')
train = pd.read_pickle('data/train.pkl')
test = pd.read_pickle('data/test_B.pkl')

drop_cols = ['label', 'query_id', 'doc_id']
feats = train.columns.drop(drop_cols).values.tolist()
X_test = test[feats]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
oof = np.zeros(train.shape[0])
sub = np.zeros(test.shape[0])

print('start training')
for i, (trn_idx, val_idx) in enumerate(skf.split(train, train.label)):
    print('----------------------{} fold----------------------'.format(i))
    X_trn, Y_trn = train.iloc[trn_idx][feats], train.iloc[trn_idx].label
    X_val, Y_val = train.iloc[val_idx][feats], train.iloc[val_idx].label
    
    clf = LGBMRegressor(
        n_estimators=100000,
        learning_rate=0.1,
        num_leaves=255,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2020,
        metric='RMSE',
        n_jobs=24,
    )
    clf.fit(
        X_trn, Y_trn,
        eval_set = [(X_val, Y_val)],
        early_stopping_rounds=200,
        verbose=1000,
    )
    oof[val_idx] = clf.predict(X_val)
    sub += clf.predict(X_test) / skf.n_splits
    
sub = pd.DataFrame({
    'queryid': test.query_id,
    'documentid': test.doc_id,
    'predict_label': sub,
})

oof = pd.DataFrame({
    'query_id': train.query_id,
    'doc_id': train.doc_id,
    'oof': oof,
    'label': train.label,
})

sub.to_csv('sub_lgb_rmse.csv', index=False)
oof.to_pickle('oof_lgb_rmse.pkl')

print('finish training')