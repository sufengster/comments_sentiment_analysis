# -*- coding: utf-8 -*-

from sklearn.linear_model import SGDClassifier
import numpy as np
import pandas as pd
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


def trainClassifer(train_vecs, y_train, column_name):
    lr = SGDClassifier(loss='log', penalty='l2')
    lr.fit(train_vecs, y_train)
    joblib.dump(lr, columns_model_path+column_name+'.pkl')

for column in columns:
       print('training ' + column)
       train_vecs = np.loadtxt(columns_train_path + column + '_vecs.txt', dtype=float, delimiter=',')
       y_train = np.loadtxt(columns_train_path + column + '_predicts.txt')
       trainClassifer(train_vecs, y_train, column)