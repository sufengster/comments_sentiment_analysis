# -*- coding: utf-8 -*-

import pandas as pd
import os

columns = ['location_traffic_convenience',
       'location_distance_from_business_district', 'location_easy_to_find',
       'service_wait_time', 'service_waiters_attitude',
       'service_parking_convenience', 'service_serving_speed', 'price_level',
       'price_cost_effective', 'price_discount', 'environment_decoration',
       'environment_noise', 'environment_space', 'environment_cleaness',
       'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
       'others_overall_experience', 'others_willing_to_consume_again']

path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v2/train/'
columns_path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v2/columns_train/'

df_train = pd.DataFrame(pd.read_csv(path+'sentiment_analysis_trainingset_fenci.csv', header=0))
df_train.set_index('id')


def build_fastext_trainset(column):
    # os.remove(columns_path+column+'.txt')
    print(column)
    train_file = open(columns_path + column + '.txt', 'w')
    df_train[['content',column]].to_csv(columns_path+column+'.txt', header=0, index=0, sep=',')
    for i in range(len(df_train['id'])):
        train_file.write('%s__label__%d\n' % (df_train.loc[i,'content'].replace('"',''), df_train.loc[i, column]))


for column in columns:
    build_fastext_trainset(column)


