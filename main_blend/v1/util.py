import numpy as np
import jieba
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale

columns = ['location_traffic_convenience',
       'location_distance_from_business_district', 'location_easy_to_find',
       'service_wait_time', 'service_waiters_attitude',
       'service_parking_convenience', 'service_serving_speed', 'price_level',
       'price_cost_effective', 'price_discount', 'environment_decoration',
       'environment_noise', 'environment_space', 'environment_cleaness',
       'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
       'others_overall_experience', 'others_willing_to_consume_again']

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

qinggans_path = '/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/dict/qinggan/'
qinggans = {}
for column in columns:
    column_qinggans = { (i.strip())  for i in open(qinggans_path + column + '.txt','r').readlines() }
    qinggans[column] = column_qinggans

whole_qinggans = {(i.strip())  for i in open(qinggans_path + 'total_qingganci.txt','r').readlines()}
qinggans['whole'] = whole_qinggans

# for word in qinggans['location_traffic_convenience']:
#     print(word)
#
# if '可以' in qinggans['location_traffic_convenience']:
#     print('yes')

def getQingganString(text, column):
    lines =  text.split('\n')
    ret = ''
    for line in lines:
        for word in line.split(' '):
            if word in qinggans[column]:
                ret += line
                ret += ' '
                break
    return ret.strip()

# text = ''' 榴莲 酥 榴莲 味道 不足 松软 奶味 浓
#  虾饺 好吃 两颗 大 虾仁
#  皮蛋 粥 皮蛋 多 但是 一般 挺 稠
#  奶黄包 很 好吃 真的 蛋黄 奶 而且 真的 流沙
#  叉烧包 面香
#  鲜虾 烧卖 好吃 外面 黄色 皮 看着 特别 食欲
#  云吞面 云吞 分量 足 但是 汤头 不是 很 好喝 而且 云吞 馅儿 不知 为何 感觉 不是 很 新鲜
#  鲍汁 腐皮卷 没 怎么 吃 味道 不错
#  排骨 味道 不错 不算 很腻 但是 油 确实 微多
#  鲜虾 锅贴 确实 今天 吃 很多 这个 很 酥脆
#  里头 很 好吃
#  刚好 优惠券 '''
# print(getQingganString(text, 'service_waiters_attitude'))


def buildRowVector(text, word2vec_size, word2vec_model, column):
    word2vec_arr = buildWordVector(text, word2vec_size, word2vec_model)
    # qinggan whole
    vec_whole_arr = buildWordVector(getQingganString(text, 'whole'), word2vec_size, word2vec_model)
    # qinggan for column
    vec_column_arr = buildWordVector(getQingganString(text, column), word2vec_size, word2vec_model)

    array = word2vec_arr
    array = np.append(word2vec_arr, vec_whole_arr)
    array = np.append(array, vec_column_arr)

    # print('size',array.size)
    # return np.asmatrix(array)
    return array


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