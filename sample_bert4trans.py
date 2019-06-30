import requests
import json
import pickle

import pandas as pd
import numpy as np
import math

query_record_train = pd.read_csv('../kdd_phase2/data_set_phase2/train_queries.csv')
label = pd.read_csv('../kdd_phase2/data_set_phase2/train_clicks.csv')

label.columns=['sid','timestamp','transport_mode']
label.drop('timestamp',axis=1,inplace=True)

query_record_train = pd.merge(query_record_train,label,on=['sid'],how='left').fillna(0)
query_record_test = pd.read_csv('../kdd_phase2/data_set_phase2/test_queries.csv')
query_record = pd.concat([query_record_train,query_record_test],axis=0,ignore_index=True)

query_record.columns = ['d', 'o', 'pid', 'timestamp', 'sid', 'transport_mode']
query_record = query_record[['sid','pid', 'timestamp','d', 'o', 'transport_mode']]

query_record = query_record.merge(pd.read_csv('../kdd_phase2/temp_data/geo_o.csv').drop_duplicates()[['o','geo_o_5']],on='o',how='left')
query_record = query_record.merge(pd.read_csv('../kdd_phase2/temp_data/geo_d.csv').drop_duplicates()[['d','geo_d_5']],on='d',how='left')

vocab = query_record['geo_o_5'].drop_duplicates()
vocab2 = query_record['geo_d_5'].drop_duplicates()

pd.concat([vocab,vocab2],axis=0).drop_duplicates().to_csv('vocab.txt',index=False,header=None)

res = query_record.groupby('pid')['geo_text'].apply((lambda x :' '.join(x))).reset_index()
res = query_record.groupby('pid')['geo_text'].apply((lambda x :' '.join(x))).reset_index()
res['geo_text'].to_csv('geo_5_text.txt',',index=False,header=None)

