# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from IPython.display import display
import jieba

path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/train/'

training_file = path+'sentiment_analysis_trainingset_fenci.csv'
df = pd.DataFrame(pd.read_csv(training_file,header=0))
df.set_index('id')


df['content'].to_csv(path+"trainset_contents_fenci.txt", index=0)