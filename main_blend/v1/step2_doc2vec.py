# -*- coding: utf-8 -*-
import gensim
import jieba
import pandas as pd
from gensim.models.doc2vec import TaggedDocument

path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/train/'
model_path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/model/'


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield TaggedDocument(words=doc.split(), tags=[self.labels_list[idx]])

df_train = pd.DataFrame(pd.read_csv(path+'sentiment_analysis_trainingset_fenci.csv', header=0))
df_train.set_index('id')

print('build train set done...')

it = LabeledLineSentence(df_train['content'], df_train['id'])

print('build LabeledLineSentence done...')

model = gensim.models.Doc2Vec(size=100, window=10, min_count=5, workers=11, alpha=0.025, min_alpha=0.025)
model.build_vocab(it)
for epoch in range(2):
    model.train(it, total_examples=df_train.size, epochs=model.epochs)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no deca
    model.train(it, total_examples=df_train.size, epochs=model.epochs)

model.save(model_path+'contents_doc2vec.model')







