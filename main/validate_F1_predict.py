# -*- coding: utf-8 -*-

from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd

from main.apply_df_by_multiprocessing import apply_by_multiprocessing
from test.util import buildWordVector, trainClassifer
import gensim
import joblib
from sklearn.metrics import f1_score
import copy

columns = ['location_traffic_convenience',
       'location_distance_from_business_district', 'location_easy_to_find',
       'service_wait_time', 'service_waiters_attitude',
       'service_parking_convenience', 'service_serving_speed', 'price_level',
       'price_cost_effective', 'price_discount', 'environment_decoration',
       'environment_noise', 'environment_space', 'environment_cleaness',
       'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
       'others_overall_experience', 'others_willing_to_consume_again']

# generate predict result file 'validate_result_file'
validate_file = '../data/validate/sentiment_analysis_validationset.csv'
validate_result_file = '../data/validate/sentiment_analysis_validationset_result.csv'

df_validate = pd.DataFrame(pd.read_csv(validate_file, header=0))
df_validate.set_index('id')

df_validate_result = copy.deepcopy(df_validate)

print(id(df_validate))
print(id(df_validate_result))

model = gensim.models.Word2Vec.load('../data/model/contents_word2vec.model')

lrs = {}
for column in columns:
    lrs[column] = joblib.load('../data/model/classifier_'+column+'.pkl')

def predict4row(row):
    index = row['id']
    print(index)
    vec = buildWordVector(row['content'], 100, model)
    for column in columns:
        predict = lrs[column].predict(vec)
        df_validate_result.loc[index, column] = predict[0]
       # print('len: %d %s %d' % (index,column,df_test.loc[index, column]))

apply_by_multiprocessing(df_validate_result, predict4row, axis=1, workers=8)

df_validate_result.to_csv(validate_result_file, index=0,float_format='%.0f')

for column in columns:
       score = f1_score(df_validate[column], df_validate_result[column],average='macro')
       print('%s f1 score: %f' % (column, score))


