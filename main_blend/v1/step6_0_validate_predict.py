# -*- coding: utf-8 -*-

import copy

import gensim
import joblib
import pandas as pd
from sklearn.metrics import f1_score

from main_blend.apply_df_by_multiprocessing import apply_by_multiprocessing
from main_blend.v1.util import buildRowVector

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

# generate predict result file 'validate_result_file'
validate_file = path+'sentiment_analysis_validationset_fenci.csv'
validate_result_file = path+'sentiment_analysis_validationset_fenci_result.csv'

df_validate = pd.DataFrame(pd.read_csv(validate_file, header=0))
df_validate.set_index('id')

df_validate_result = copy.deepcopy(df_validate)
df_validate_result.set_index('id')

print(df_validate_result.shape)

word2vec_model = gensim.models.Word2Vec.load(model_path+'contents_word2vec.model')
doc2vec_model = gensim.models.Doc2Vec.load(model_path+'contents_doc2vec.model')

lrs = {}
for column in columns:
    lrs[column] = joblib.load(columns_model_path+column+'.pkl')

def predict4row(row):
    index = row['id']
    vec = buildRowVector(row['content'], 100, 100, word2vec_model, doc2vec_model)
    for column in columns:
        predict = lrs[column].predict(vec)
        predict_proba = lrs[column].predict_proba(vec)
        print('===================')
        print('proba: ', predict_proba[0])
        print('predict: ', predict[0])
        print('real: ', df_validate_result.loc[index, column])
        # print('   %d, %s, %d, %d' % (index, column, df_validate_result.loc[index, column], predict[0]))
        df_validate_result.loc[index, column] = int(predict[0])

apply_by_multiprocessing(df_validate_result, predict4row, axis=1, workers=1)

df_validate_result.to_csv(validate_result_file, index=0,float_format='%.0f')

total = 0
for column in columns:
       score = f1_score(df_validate[column], df_validate_result[column],average='macro')
       print('%s f1 score: %f' % (column, score))
       total += score

print(' F1 Score: %f'%(total/20))


