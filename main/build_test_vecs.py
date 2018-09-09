# -*- coding: utf-8 -*-

from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
from test.util import buildWordVector
import gensim
import joblib

test_a_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/test_data_a/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv'

df_test = pd.DataFrame(pd.read_csv(test_a_file, header=0))
df_test.set_index('id')

test_vecs = np.concatenate([buildWordVector(z, 100, model) for z in df_test['content']])
test_vecs = scale(validate_vecs)
np.savetxt('../data/model/validate_vecs.txt', validate_vecs, fmt='%f', delimiter=',')