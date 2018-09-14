# -*- coding: utf-8 -*-
import joblib
import nltk
import jieba
import pandas as pd

# nltk.download()
# train_file = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/train/sentiment_analysis_trainingset_fenci.csv'
#
# df = pd.DataFrame(pd.read_csv(train_file,header=0))
# df.set_index('id')



# raw=open('/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/train/trainset_contents_fenci.txt').read()
# text=nltk.text.Text(raw.replace('"""','').split())
#
# joblib.dump(text, '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/dict/nltktext')
text = joblib.load('/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/dict/nltktext')


print(text.__len__())
print(text.count('好吃'))
# print(text.concordance('好吃'))
# print(text.dispersion_plot('好吃'))
print(text.collocations(num=10000, window_size=5))