# -*- coding: utf-8 -*-
from copy import deepcopy

import gensim
import numpy as np
import pandas as pd
import time
import os

from main_blend.v1.util import buildRowVector

columns = ['location_traffic_convenience',
           'location_distance_from_business_district', 'location_easy_to_find',
           'service_wait_time', 'service_waiters_attitude',
           'service_parking_convenience', 'service_serving_speed', 'price_level',
           'price_cost_effective', 'price_discount', 'environment_decoration',
           'environment_noise', 'environment_space', 'environment_cleaness',
           'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
           'others_overall_experience', 'others_willing_to_consume_again']

values = [-2, 1, 0, -1]

path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/train/'
columns_train_path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/columns_train/'
model_path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/model/'

df_train = pd.DataFrame(pd.read_csv(path + 'sentiment_analysis_trainingset_fenci.csv', header=0))
df_train.set_index('id')

# train_vecs = np.loadtxt(model_path + 'train_vecs_noscale.txt', dtype=float, delimiter=',')
word2vec_model = gensim.models.Word2Vec.load(model_path+'contents_word2vec.model')

for column in columns:
    print('start creating trainset for : ', column)
    starttime = time.time()
    # group = df_train.groupby(column).id.count()
    # maxLine = max([group[1], group[0], group[-1], group[-2]])
    # times_1 = round(maxLine / group[1])
    # times_0 = round(maxLine / group[0])
    # times_m1 = round(maxLine / group[-1])
    # times_m2 = round(maxLine / group[-2])
    #
    # print(times_1)
    # print(times_0)
    # print(times_m1)
    # print(times_m2)
    #
    # times_map = {}
    # times_map[1] = int(times_1)
    # times_map[0] = int(times_0)
    # times_map[-1] = int(times_m1)
    # times_map[-2] = int(times_m2)

    print('length: ',len(df_train[column]))

    os.remove(columns_train_path + column + '_vecs.txt')
    os.remove(columns_train_path + column + '_predicts.txt')

    vecs_file = open(columns_train_path + column + '_vecs.txt', 'w')
    predicts_file = open(columns_train_path + column + '_predicts.txt', 'w')

    for i in range(len(df_train[column])):
        print(i)
        # build vector for each row
        vector = buildRowVector(df_train['content'][i], 100, word2vec_model, column)
        class_value = int(df_train[column][i])
        vecs_file.write('%s\n' % ','.join('{:.6f}'.format(n) for n in vector))
        predicts_file.write('%d\n' % class_value)

    print(' cost time ', (time.time()-starttime))