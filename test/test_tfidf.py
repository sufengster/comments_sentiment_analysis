# -*- coding: utf-8 -*-
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/train/'
df_train = pd.DataFrame(pd.read_csv(path + 'sentiment_analysis_trainingset_fenci.csv', header=0))
df_train.set_index('id')

corpus = [
    'This is the first document.',
    'This is the second document.',
    'And the third one',
    'Is this the first document?',
    'I come to American to travel'
]
count_vec = CountVectorizer(stop_words=None)
words = count_vec.fit_transform(df_train['content'].values)
tfidf = TfidfVectorizer().fit_transform(words)



print(tfidf.vocabulary_)
# for key, value in words.vocabulary_.items():
#     print(key, value)
# print (words.vocabulary_['好吃'])
