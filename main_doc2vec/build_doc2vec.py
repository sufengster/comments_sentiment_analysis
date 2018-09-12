# -*- coding: utf-8 -*-
import jieba
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
from gensim import utils
from gensim.models.doc2vec import LabeledSentence, TaggedDocument
from gensim.models import Doc2Vec


columns = ['location_traffic_convenience',
       'location_distance_from_business_district', 'location_easy_to_find',
       'service_wait_time', 'service_waiters_attitude',
       'service_parking_convenience', 'service_serving_speed', 'price_level',
       'price_cost_effective', 'price_discount', 'environment_decoration',
       'environment_noise', 'environment_space', 'environment_cleaness',
       'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
       'others_overall_experience', 'others_willing_to_consume_again']

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            doc_terms = ' '.join(jieba.cut(doc, cut_all=False))
            yield TaggedDocument(words=doc_terms.split(), tags=[self.labels_list[idx]])





training_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/training_data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'
validate_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/verify_data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv'
test_a_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/test_data_a/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv'

df_train = pd.DataFrame(pd.read_csv(validate_file, header=0))
df_train.set_index('id')


def train_column(column):
    print('start training %s'%column)
    it = LabeledLineSentence(df_train['content'], df_train['id'])
    model = gensim.models.Doc2Vec(size=300, window=10, min_count=5, workers=11, alpha=0.025, min_alpha=0.025)
    model.build_vocab(it)
    print('model build done for %s' % column)
    for epoch in range(10):
        model.train(it, total_examples=df_train.size, epochs=model.epochs)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no deca
        model.train(it, total_examples=df_train.size, epochs=model.epochs)

    model.save('../data/model_doc2vec/%s_doc2vec.model'%column)

for column in columns:
    train_column(column)







