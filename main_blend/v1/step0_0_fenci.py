# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from IPython.display import display
import jieba

from main_blend.v1.util import seg_sentence
import re

training_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/training_data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'
validate_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/verify_data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv'
test_a_file = '/Users/sufeng/AIChallenger/细粒度用户评论情感分析/test_data_a/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv'


path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/train/'

def fenci(filename):
    df = pd.DataFrame(pd.read_csv(path+filename+'.csv',header=0))
    df.set_index('id')

    for i in range(0, len(df)):
        content = df.loc[i,'content']
        content_fenci = seg_sentence(re.sub('\n{2,}','\n',re.sub('[，.,，。!！~]', '\n', content)))
        df.loc[i,'content'] = content_fenci

    df.to_csv(path+filename+'_fenci.csv', index=0)
    print('fenci done for %s' % filename)

fenci('sentiment_analysis_trainingset')
fenci('sentiment_analysis_validationset')
fenci('sentiment_analysis_testa')