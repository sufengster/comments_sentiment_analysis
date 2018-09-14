# -*- coding: utf-8 -*-

import jieba

from main_blend.v1.util import seg_sentence

# jieba.analyse.set_stop_words('/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/dict/stopwords.txt')
# jieba.load_userdict('/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/dict/user_dict.txt')


string = '由于是饿到一定程度了，所以我觉得什么都好吃，但又因为只有2人 ，又没有点全 ， 只能以后再来尝试朋友们推荐的其他东西'

terms = seg_sentence(string)

print(terms)
print(terms.split())

