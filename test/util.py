import numpy as np
import jieba

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