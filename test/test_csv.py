# -*- coding: utf-8 -*-

import csv
path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/train/'

training_file = path+'sentiment_analysis_trainingset_fenci.csv'

with open(training_file, mode='r') as f:
    reader = csv.reader(f)
    linenumber = 1
    try:
        for row in reader:
            linenumber += 1
    except Exception as e:
        print (("Error line %d: %s %s" % (linenumber, str(type(e)), e)))