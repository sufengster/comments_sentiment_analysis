# -*- coding: utf-8 -*-

from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
from test.util import buildWordVector, trainClassifer
import gensim
import joblib
import time

columns = ['location_traffic_convenience',
       'location_distance_from_business_district', 'location_easy_to_find',
       'service_wait_time', 'service_waiters_attitude',
       'service_parking_convenience', 'service_serving_speed', 'price_level',
       'price_cost_effective', 'price_discount', 'environment_decoration',
       'environment_noise', 'environment_space', 'environment_cleaness',
       'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
       'others_overall_experience', 'others_willing_to_consume_again']

test_a_file = '../data/result/sentiment_analysis_testa.csv'
test_a_file_result = '../data/result/sentiment_analysis_testa_result.bak.csv'

df_test = pd.DataFrame(pd.read_csv(test_a_file, header=0))
df_test.set_index('id')

model = gensim.models.Word2Vec.load('../data/model/contents_word2vec.model')

lrs = {}
for column in columns:
    lrs[column] = joblib.load('../data/model/classifier_'+column+'.pkl')

for i in range(0, len(df_test)):
    start_time = time.time()
    print(i)
    # print(df_test.iloc[i]['content'], df_test.iloc[i]['location_distance_from_business_district'])
    vec = buildWordVector(df_test.iloc[i]['content'], 100, model)
    print("buildWordVector --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    for column in columns:
        predict = lrs[column].predict(vec)
        df_test.iloc[i][column] = predict[0]
        # print(column+': '+predict[0])
        # print(predict[0])
    print("predict --- %s seconds ---" % (time.time() - start_time))

df_test.to_csv(test_a_file_result)