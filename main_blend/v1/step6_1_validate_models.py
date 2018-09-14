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


path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/train/'
columns_train_path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/columns_train/'
columns_model_path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/columns_model/'
model_path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/model/'

df_validate = pd.DataFrame(pd.read_csv(validate_file, header=0))
df_validate.set_index('id')

print(df_validate.dtypes)

validate_vecs = np.loadtxt('../data/model/validate_vecs.txt', dtype=float, delimiter=',')

for column in columns:
       y_validate = df_validate[column]
       lr = joblib.load('../data/model/classifier_'+column+'.pkl')
       print(column+ ' validate accuracy: %.2f' % lr.score(validate_vecs, y_validate))