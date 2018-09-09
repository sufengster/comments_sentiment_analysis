# -*- coding: utf-8 -*-
import threading

from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd

from main.apply_df_by_multiprocessing import apply_by_multiprocessing
from test.util import buildWordVector, trainClassifer
import gensim
import joblib
import time
from multiprocessing.dummy import Pool as ThreadPool

columns = ['location_traffic_convenience',
       'location_distance_from_business_district', 'location_easy_to_find',
       'service_wait_time', 'service_waiters_attitude',
       'service_parking_convenience', 'service_serving_speed', 'price_level',
       'price_cost_effective', 'price_discount', 'environment_decoration',
       'environment_noise', 'environment_space', 'environment_cleaness',
       'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
       'others_overall_experience', 'others_willing_to_consume_again']

test_a_file = '../data/result/sentiment_analysis_testa.csv'
test_a_file_result = '../data/result/sentiment_analysis_testa_result.csv'

df_test = pd.DataFrame(pd.read_csv(test_a_file, header=0))
df_test.set_index('id')
df_test = df_test.fillna(0)

print(df_test.dtypes)

model = gensim.models.Word2Vec.load('../data/model/contents_word2vec.model')

lrs = {}
for column in columns:
    lrs[column] = joblib.load('../data/model/classifier_'+column+'.pkl')
    df_test[column] = df_test[column].astype(int)

# for i in range(0, len(df_test)):
#     print(i)
#     vec = buildWordVector(df_test.iloc[i]['content'], 100, model)
#     for column in columns:
#         predict = lrs[column].predict(vec)
#         df_test.iloc[i][column] = predict[0]

def predict4row(row):
    # global df_test
    index = row['id']
    print(index)
    vec = buildWordVector(row['content'], 100, model)
    for column in columns:
        predict = lrs[column].predict(vec)
        df_test.loc[index, column] = predict[0]
        print('len: %d %s %d %d' % (index,column,df_test.loc[index, column], id(df_test)))

print('value1: %s %d' % (df_test.loc[1,'price_discount'], id(df_test)))
apply_by_multiprocessing(df_test, predict4row, axis=1, workers=6)
print('value2: %s %d' % (df_test.loc[1,'price_discount'], id(df_test)))


df_test.to_csv(test_a_file_result, index=0)