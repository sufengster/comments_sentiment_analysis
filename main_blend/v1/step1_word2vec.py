# -*- coding: utf-8 -*-

import gensim
import jieba
from gensim.models import word2vec

path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/train/'
model_path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/model/'
word2vec_trainset = path+"trainset_contents_fenci.txt"

sentences = word2vec.Text8Corpus(word2vec_trainset)

model = gensim.models.Word2Vec(sentences, size=300, min_count=200, workers=6)
model.save(model_path+'contents_word2vec.model')