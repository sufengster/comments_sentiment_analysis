import jieba
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier


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


vec1 = np.zeros(10).reshape((1, 10))
vec2 = np.ones(10).reshape((1, 10))

vec = np.concatenate([vec1,vec2])

print(vec1)
print(vec2)
print(vec)

print(np.append(vec1[0], vec2[0]))