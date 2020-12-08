import pandas as pd
import numpy as np
from tqdm import tqdm
from lightgbm.sklearn import LGBMRanker
from sklearn.model_selection import StratifiedKFold

oof_lgb_rank = pd.read_pickle('oof_lgb_rank.pkl').sort_values(['query_id', 'doc_id'], ignore_index=True)
oof_lgb_rmse = pd.read_pickle('oof_lgb_rmse.pkl').sort_values(['query_id', 'doc_id'], ignore_index=True)
oof_xgb_rank = pd.read_pickle('oof_xgb_rank.pkl').sort_values(['query_id', 'doc_id'], ignore_index=True)
oof_xgb_rmse = pd.read_pickle('oof_xgb_rmse.pkl').sort_values(['query_id', 'doc_id'], ignore_index=True)

sub_lgb_rank = pd.read_csv('sub_lgb_rank.csv').sort_values(['queryid', 'documentid'], ignore_index=True)
sub_lgb_rmse = pd.read_csv('sub_lgb_rmse.csv').sort_values(['queryid', 'documentid'], ignore_index=True)
sub_xgb_rank = pd.read_csv('sub_xgb_rank.csv').sort_values(['queryid', 'documentid'], ignore_index=True)
sub_xgb_rmse = pd.read_csv('sub_xgb_rmse.csv').sort_values(['queryid', 'documentid'], ignore_index=True)

train = pd.concat([oof_lgb_rank, oof_lgb_rmse.oof, oof_xgb_rank.oof, oof_xgb_rmse.oof], axis=1)
test = pd.concat([sub_lgb_rank, sub_lgb_rmse.predict_label, sub_xgb_rank.predict_label, sub_xgb_rmse.predict_label], axis=1)

train.columns = ['query_id', 'doc_id', 'x_0', 'label', 'x_1', 'x_2', 'x_3']
test.columns = ['queryid', 'documentid', 'x_0', 'x_1', 'x_2', 'x_3']

feats = ['x_0', 'x_1', 'x_2', 'x_3']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
X_test = test[feats]
sub = pd.DataFrame({
    'queryid': test.queryid,
    'documentid': test.documentid,
    'predict_label': 0.0,
})

for i, (trn_idx, val_idx) in enumerate(skf.split(train, train.label)):
    print('----------------------{} fold----------------------'.format(i))
    trn_df = train.iloc[trn_idx].sort_values('query_id', ignore_index=True)
    val_df = train.iloc[val_idx].sort_values('query_id', ignore_index=True)
    X_trn, Y_trn = trn_df[feats], trn_df.label.values
    group_trn = trn_df.query_id.value_counts().sort_index().values
    X_val, Y_val = val_df[feats], val_df.label.values
    group_val = val_df.query_id.value_counts().sort_index().values
    
    ranker = LGBMRanker(
        num_leaves=31,
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
        verbose=200,
    )
    
    sub['predict_label'] += ranker.predict(X_test) / skf.n_splits

sub.to_csv('submission.csv', index=False)