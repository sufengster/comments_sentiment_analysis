# -*- coding: utf-8 -*-

import nltk
import jieba
import pandas as pd

# nltk.download()
train_file = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/train/sentiment_analysis_trainingset_fenci.csv'

df = pd.DataFrame(pd.read_csv(train_file,header=0))
df.set_index('id')

text=nltk.text.Text(' '.join(df['content']).split())

print(text.__len__())
print(text.count('好吃'))
# print(text.concordance('好吃'))
# print(text.dispersion_plot('好吃'))
print(text.collocations(num=1000, window_size=2))