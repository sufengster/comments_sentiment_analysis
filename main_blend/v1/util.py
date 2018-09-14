import numpy as np
import jieba
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale

def buildWordVector(text, size, model):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in text.split():
        try:
            vec += model[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec[0]

def buildDocVector(text, size, model):
    try:
        arr = model.infer_vector(text.split())
    except KeyError:
        print('buildDocVector error: %s' % text)
    return arr

# def trainClassifer(train_vecs, y_train, column_name):
#     lr = SGDClassifier(loss='log', penalty='l2')
#     lr.fit(train_vecs, y_train)
#     joblib.dump(lr, '../data/model/classifier_'+column_name+'.pkl')
jieba.load_userdict('/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/dict/user_dict.txt')
stopwords = [line.strip() for line in open('/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/dict/stopwords.txt').readlines() ]


def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t' and word !=' ':
                outstr += word
                outstr += " "
    return outstr


def buildRowVector(text, word2vec_size, doc2vec_size, word2vec_model, doc2vec_model):
    word2vec_arr = buildWordVector(text, word2vec_size, word2vec_model)
    # doc2vec_arr = buildDocVector(text, doc2vec_size, doc2vec_model)
    array = word2vec_arr
    # array = np.append(word2vec_arr, doc2vec_arr)
    # array = np.append(array, [len(text)])
    # array = np.append(array, [len(text.split())])

    return np.asmatrix(array)


# arr1 = [0,0,0]
# arr2 = [1,1,1]
#
# array = np.append(arr1, arr2)
# array = np.append(array, [2])
# array = np.append(array, [3])
#
# vec = np.asmatrix(array)
#
# print(array)
# print(vec)