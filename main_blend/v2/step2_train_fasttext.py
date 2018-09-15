# -*- coding: utf-8 -*-

import logging

import fasttext

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

columns = ['location_traffic_convenience',
       'location_distance_from_business_district', 'location_easy_to_find',
       'service_wait_time', 'service_waiters_attitude',
       'service_parking_convenience', 'service_serving_speed', 'price_level',
       'price_cost_effective', 'price_discount', 'environment_decoration',
       'environment_noise', 'environment_space', 'environment_cleaness',
       'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
       'others_overall_experience', 'others_willing_to_consume_again']

column_model_path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v2/columns_model/'
columns_path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v2/columns_train/'


for column in columns:
       classifier = fasttext.supervised(input_file=columns_path+column+".txt", output=column_model_path+column+".model", label_prefix="__label__",dim=200, min_count=100, lr=1.0, epoch=25)
