# -*- coding: utf-8 -*-

import gensim
import jieba
from gensim.models import word2vec

model = gensim.models.Word2Vec.load('../data/model/contents_word2vec.model')

vector = model['好吃']

print(vector)

req_count = 15
for key in model.similar_by_word(u'好吃',topn =100):
    if len(key[0])==3:
        req_count -=1
        print(key[0],key[1])
        if req_count ==0:
            break

req_count = 15
for key in model.similar_by_word(u'恶略',topn =100):
    if len(key[0])==3:
        req_count -=1
        print(key[0],key[1])
        if req_count ==0:
            break

req_count = 15
for key in model.similar_by_word(u'很近',topn =100):
    if len(key[0])==3:
        req_count -=1
        print(key[0],key[1])
        if req_count ==0:
            break