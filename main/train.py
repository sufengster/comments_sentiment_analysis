# -*- coding: utf-8 -*-

from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
from test.util import buildWordVector, trainClassifer
import gensim
import joblib

columns = ['location_traffic_convenience',
       'location_distance_from_business_district', 'location_easy_to_find',
       'service_wait_time', 'service_waiters_attitude',
       'service_parking_convenience', 'service_serving_speed', 'price_level',
       'price_cost_effective', 'price_discount', 'environment_decoration',
       'environment_noise', 'environment_space', 'environment_cleaness',
       'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
       'others_overall_experience', 'others_willing_to_consume_again']


training_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/training_data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'
validate_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/verify_data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv'
test_a_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/test_data_a/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv'

df_train = pd.DataFrame(pd.read_csv(training_file, header=0))
df_train.set_index('id')

train_vecs = np.loadtxt('../data/model/train_vecs.txt', dtype=float, delimiter=',')

for column in columns:
       print('training ' + column)
       y_train = df_train[column]
       trainClassifer(train_vecs, y_train, column)