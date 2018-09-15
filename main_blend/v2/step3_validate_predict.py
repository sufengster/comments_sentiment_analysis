# -*- coding: utf-8 -*-

import copy

import gensim
import joblib
import pandas as pd
from sklearn.metrics import f1_score

from main_blend.apply_df_by_multiprocessing import apply_by_multiprocessing
from main_blend.v1.util import buildRowVector
import fasttext

columns = ['location_traffic_convenience',
       'location_distance_from_business_district', 'location_easy_to_find',
       'service_wait_time', 'service_waiters_attitude',
       'service_parking_convenience', 'service_serving_speed', 'price_level',
       'price_cost_effective', 'price_discount', 'environment_decoration',
       'environment_noise', 'environment_space', 'environment_cleaness',
       'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
       'others_overall_experience', 'others_willing_to_consume_again']

path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/train/'
column_model_path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v2/columns_model/'

# generate predict result file 'validate_result_file'
validate_file = path+'sentiment_analysis_validationset_fenci.csv'
# validate_result_file = path+'sentiment_analysis_validationset_fenci_result.csv'

df_validate = pd.DataFrame(pd.read_csv(validate_file, header=0))
df_validate.set_index('id')

df_validate_result = copy.deepcopy(df_validate)
df_validate_result.set_index('id')

print(df_validate_result.shape)

lrs = {}
for column in columns:
    lrs[column] = fasttext.load_model(column_model_path+column+'.model.bin', label_prefix='__label__')

def predict4row(row):
    index = row['id']
    for column in columns:
        lables = lrs[column].predict([row['content'].replace('"','')])
        # print('labels: ', lables)
        #
        # print('===================')
        # print('predict: ', lables[0][0])
        # print('real: ', df_validate_result.loc[index, column])

        # print('   %d, %s, %d, %d' % (index, column, df_validate_result.loc[index, column], predict[0]))
        df_validate_result.loc[index, column] = int(lables[0][0])

apply_by_multiprocessing(df_validate_result, predict4row, axis=1, workers=1)

total = 0
for column in columns:
       score = f1_score(df_validate[column], df_validate_result[column],average='macro')
       print('%s f1 score: %f' % (column, score))
       total += score

print(' F1 Score: %f'%(total/20))


