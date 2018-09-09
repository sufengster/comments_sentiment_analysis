import numpy as np
import jieba
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale

def buildWordVector(text, size, model):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in jieba.cut(text, cut_all=False):
        try:
            vec += model[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def trainClassifer(train_vecs, y_train, column_name):
    lr = SGDClassifier(loss='log', penalty='l2')
    lr.fit(train_vecs, y_train)
    joblib.dump(lr, '../data/model/classifier_'+column_name+'.pkl')