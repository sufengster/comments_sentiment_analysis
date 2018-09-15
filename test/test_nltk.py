# -*- coding: utf-8 -*-
import joblib
import nltk
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

# nltk.download()
# train_file = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/train/sentiment_analysis_trainingset_fenci.csv'
#
# df = pd.DataFrame(pd.read_csv(train_file,header=0))
# df.set_index('id')



# raw=open('/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/train/trainset_contents_fenci.txt').read()
# text=nltk.text.Text(raw.replace('"""','').split())
#
# joblib.dump(text, '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/dict/nltktext')
# text = joblib.load('/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/dict/nltktext')
#
#
# print(text.__len__())
# print(text.count('好吃'))
# # print(text.concordance('好吃'))
# # print(text.dispersion_plot('好吃'))
# print(text.collocations(num=10000, window_size=5))

# raw=open('/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/train/trainset_contents_fenci.txt').read()
# fdist = nltk.FreqDist(nltk.word_tokenize(raw))
# joblib.dump(fdist, '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/dict/nltktext.fdist')

# fdist = joblib.load('/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/dict/nltktext.fdist')
#
# commons = fdist.most_common(10000)
# print(commons)
# print(fdist.hapaxes())

# fdist.plot(30,cumulative=True)

columns = ['location_traffic_convenience',
       'location_distance_from_business_district', 'location_easy_to_find',
       'service_wait_time', 'service_waiters_attitude',
       'service_parking_convenience', 'service_serving_speed', 'price_level',
       'price_cost_effective', 'price_discount', 'environment_decoration',
       'environment_noise', 'environment_space', 'environment_cleaness',
       'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
       'others_overall_experience', 'others_willing_to_consume_again']

path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/train/'

training_file = path+'sentiment_analysis_trainingset_fenci.csv'
df = pd.DataFrame(pd.read_csv(training_file,header=0))
df.set_index('id')

# for column in columns:
#     print('%s:'%column)
#     filter = df.loc[df[column].isin([1,0,-1])]
#     filter_contents = filter['content']
#     fdist = nltk.FreqDist(nltk.word_tokenize(''.join(filter_contents)))
#     print('    ',fdist.most_common(500))

filter = df.loc[df['service_parking_convenience'].isin([1])]

total = 0
for content in filter['content']:
    total += 1
    if(total>20):
        break
    print('%d, %s' % (total, content))
