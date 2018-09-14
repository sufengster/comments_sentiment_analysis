# -*- coding: utf-8 -*-

import gensim
import jieba
from gensim.models import word2vec

path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/train/'
model_path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/model/'
word2vec_trainset = path+"trainset_contents_fenci.txt"
jieba.suggest_freq('副市长', True)

sentences = word2vec.Text8Corpus(word2vec_trainset)

model = gensim.models.Word2Vec(sentences, min_count=5, workers=6)
model.save(model_path+'contents_word2vec.model')