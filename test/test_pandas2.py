# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

columns = ['location_traffic_convenience',
       'location_distance_from_business_district', 'location_easy_to_find',
       'service_wait_time', 'service_waiters_attitude',
       'service_parking_convenience', 'service_serving_speed', 'price_level',
       'price_cost_effective', 'price_discount', 'environment_decoration',
       'environment_noise', 'environment_space', 'environment_cleaness',
       'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
       'others_overall_experience', 'others_willing_to_consume_again']

training_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/training_data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'
validate_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/verify_data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv'
test_a_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/test_data_a/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv'

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
# pd.options.display.float_format = '{:,.3f}'.format()

df = pd.DataFrame(pd.read_csv(training_file,header=0))
df.set_index('id')
# df.info()
# print(df.columns)

for column in columns:
    group = df.groupby(column).id.count()
    # print(group)
    print('  -2 : %d' % group[-2])
    print('  -1 : %d' % group[-1])
    print('  0 : %d' % group[0])
    print('  1 : %d' % group[1])
    # df.groupby(column).id.count().plot.bar(ylim=0)
    # plt.show()


# df.shape
#
# df.info()
#
# df.values
# print(df.head(5))

# head2 = df.head(2)
#
# print(head2)
#
# print(head2['content'])
# print(head2['location_traffic_convenience'])
# display(df.head(5))

# df.plot(x='id', y=['location_traffic_convenience'])

