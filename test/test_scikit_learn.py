# -*- coding: utf-8 -*-

from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
from test.util import buildWordVector
import gensim
import joblib


training_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/training_data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'
validate_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/verify_data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv'
test_a_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/test_data_a/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv'

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

print('step1')
df_train = pd.DataFrame(pd.read_csv(training_file, header=0))
df_train.set_index('id')

print('step2')
df_validate = pd.DataFrame(pd.read_csv(validate_file, header=0))
df_validate.set_index('id')

# build train data set
print('step3')
model = gensim.models.Word2Vec.load('../data/model/contents_word2vec.model')

print('step4')
train_vecs = np.concatenate([buildWordVector(z, 100, model) for z in df_train['content']])
train_vecs = scale(train_vecs)
np.savetxt('../data/model/train_vecs.txt', train_vecs, fmt='%f', delimiter=',')

print('step5')
validate_vecs = np.concatenate([buildWordVector(z, 100, model) for z in df_validate['content']])
validate_vecs = scale(validate_vecs)
np.savetxt('../data/model/validate_vecs.txt', validate_vecs, fmt='%f', delimiter=',')

print('step6')
y_train = df_train['SGDClassifier']
y_validate = df_validate['dish_taste']

# train lr
print('step7')
lr = SGDClassifier(loss='log', penalty='l2')
lr.fit(train_vecs, y_train)
joblib.dump(lr, '../data/classifier_dish_taste.pkl')


print('step8')
print('Test Accuracy: %.2f' % lr.score(validate_vecs, y_validate))
