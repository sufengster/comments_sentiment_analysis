# -*- coding: utf-8 -*-
import gensim
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

from main_blend.v1.util import buildRowVector

path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/train/'
model_path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/model/'

df_train = pd.DataFrame(pd.read_csv(path+'sentiment_analysis_trainingset_fenci.csv', header=0))
df_train.set_index('id')

word2vec_model = gensim.models.Word2Vec.load(model_path+'contents_word2vec.model')
# doc2vec_model = gensim.models.Doc2Vec.load(model_path+'contents_doc2vec.model')


train_vecs = np.concatenate([buildRowVector(z, 100, word2vec_model ) for z in df_train['content']])
# scale or not? 不能scale！！！
# train_vecs = scale(train_vecs)
np.savetxt(model_path+'train_vecs_noscale.txt', train_vecs, fmt='%f', delimiter=',')







