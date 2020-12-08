import pandas as pd
import os

from utils import *
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('data/train_dataset.csv', sep='\t', names=['label','query_id','doc_id'] + ['f_{}'.format(i) for i in range(362)])
test_A = pd.read_csv('data/test_dataset_A.csv', sep='\t', names=['query_id','doc_id'] + ['f_{}'.format(i) for i in range(362)])
test_B = pd.read_csv('data/test_dataset_B.csv', sep='\t', names=['query_id','doc_id'] + ['f_{}'.format(i) for i in range(362)])

train['query_id'] = LabelEncoder().fit_transform(train['query_id'])
train['doc_id'] = LabelEncoder().fit_transform(train['doc_id'])

reduce_mem(train).to_pickle('data/train.pkl')
reduce_mem(test_A).to_pickle('data/test_A.pkl')
reduce_mem(test_B).to_pickle('data/test_B.pkl')