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

# read original & predict result file 'validate_result_file'
validate_file = '../data/validate/sentiment_analysis_validationset.csv'
validate_result_file = '../data/validate/sentiment_analysis_validationset_result.csv'

df_validate = pd.DataFrame(pd.read_csv(validate_file, header=0))
df_validate.set_index('id')

df_validate_result = pd.DataFrame(pd.read_csv(validate_result_file, header=0))
df_validate_result.set_index('id')


def f1_score_mean():
    total = 0
    for column in columns:
        score = f1_score(df_validate[column], df_validate_result[column], average='macro')
        print('%s f1 score: %f' % (column, score))
        total += score
    return total/20

print('f1_score_mean: %f' % f1_score_mean())