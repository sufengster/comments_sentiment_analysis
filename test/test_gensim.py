# -*- coding: utf-8 -*-

import gensim
import jieba
from gensim.models import word2vec

jieba.suggest_freq('副市长', True)

with open('../data/test/contents.txt',
          'rb') as f:
    document = f.read()
    document_cut = jieba.cut(document, cut_all=False)
    # print('/'.join(document_cut))
    result = ' '.join(document_cut)
    result = result.encode('utf-8')

    with open('../data/test/contents_segment.txt', 'wb+') as f1:
        f1.write(result)  # 读取的方式和写入的方式要一致
f.close()
f1.close()

print('step1...')

sentences = word2vec.Text8Corpus(r'../data/test/contents_segment.txt')

model = gensim.models.Word2Vec(sentences, min_count=5, workers=2)
model.save('../data/model/contents_word2vec.model')

print('step2...')

req_count = 5
for key in model.similar_by_word(u'好吃',topn =100):
    if len(key[0])==3:
        req_count -=1
        print(key[0],key[1])
        if req_count ==0:
            break