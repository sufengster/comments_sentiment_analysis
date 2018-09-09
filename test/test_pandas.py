# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from IPython.display import display

training_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/training_data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'
validate_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/verify_data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv'
test_a_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/test_data_a/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv'

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
# pd.options.display.float_format = '{:,.3f}'.format()

df = pd.DataFrame(pd.read_csv(training_file,header=0))
df.set_index('id')

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

df.info()
# display(df.head(5))
display(df.describe())
df['content'].to_csv("../data/test/contents.txt")

# df.plot(x='id', y=['location_traffic_convenience'])

