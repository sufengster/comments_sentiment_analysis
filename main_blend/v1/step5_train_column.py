# -*- coding: utf-8 -*-

from sklearn.linear_model import SGDClassifier, LogisticRegression
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

df_train = pd.DataFrame(pd.read_csv(path+'sentiment_analysis_trainingset_fenci.csv', header=0))
df_train.set_index('id')

def trainClassifer(train_vecs, y_train, column_name):
    # lr = SGDClassifier(loss='log', penalty='l2')
    # https://blog.csdn.net/jark_/article/details/78342644
    lr = LogisticRegression(C=1000.0, max_iter=20, random_state=0, class_weight='balanced', solver='sag', multi_class='multinomial', n_jobs=-1)
    # lr = LogisticRegression(C=1.0, random_state=0, class_weight='balanced', n_jobs=-1)

    lr.fit(train_vecs, y_train)
    joblib.dump(lr, columns_model_path+column_name+'.pkl')

for column in columns:
       print('training ' + column)
       # train_vecs = np.loadtxt(columns_train_path + column + '_vecs.txt', dtype=float, delimiter=',')
       # y_train = np.loadtxt(columns_train_path + column + '_predicts.txt')


       #  skip step4, read directly from train_vec, because logisticRegression support class_weight=balanced
       train_vecs = np.loadtxt(model_path+'train_vecs_noscale.txt', dtype=float, delimiter=',')
       y_train = df_train[column]
       trainClassifer(train_vecs, y_train, column)