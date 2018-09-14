# -*- coding: utf-8 -*-
import gensim
import jieba
import pandas as pd
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import numpy as np
from sklearn.preprocessing import scale

str = 'haha test hello'
arr = str.split()
print(arr)

print(len(arr))




# doc2vec_model = Doc2Vec.load('/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/v1/model/contents_doc2vec.model')
# doc = '服务 一般 ， 上菜 速度 倒 是 很快 ， 人 挺 多 ， 坐在 沙发 上 感觉 很 舒服'
#
# inferred_vector_dm = doc2vec_model.infer_vector(doc.split())
# print(inferred_vector_dm)
# sims = doc2vec_model.docvecs.most_similar([inferred_vector_dm], topn=10)
#
# print(sims)

# arr = [-1.116573,1.285909,0.645130,-0.311879,-0.080911,0.635355,0.343538,-2.146346,1.229915,-0.378433,0.398027,0.851674,0.812703,-0.429344,1.316212,0.085390,0.080716,0.450960,0.041497,1.159749,-0.301396,-0.707411,2.227103,0.249519,0.508497,-1.012377,0.353883,0.720215,-0.159360,1.293501,-0.758521,0.402679,-0.308364,-0.122987,0.653180,-0.123445,0.839758,0.409250,0.027441,-0.014956,-0.680263,1.636112,-0.376556,-1.208467,1.628271,-0.215402,-0.401816,-0.191132,0.687367,-0.493354,-0.060101,0.360556,-0.346484,-1.491950,0.853183,-1.113888,-1.920906,0.012633,1.836817,0.591968,-0.058694,0.069554,-1.950086,-0.554119,0.503406,0.414566,0.218802,0.068980,0.132205,-0.245572,-1.010555,-0.106716,1.586931,-1.758151,-0.644899,0.003517,-1.726202,-2.436287,-2.151654,2.576127,-0.632004,-0.769555,-0.067045,-1.059340,-0.701729,-1.360589,0.758867,-1.046157,-0.861602,0.134730,1.851888,0.405213,0.886536,0.289048,0.337468,0.493631,0.870709,0.465112,0.907659,-0.679339,-0.167231,0.256106,-0.261634,1.944882,1.651863,1.443510,0.745836,1.058047,-0.205090,-1.250487,-0.239174,-1.620906,-1.186688,-0.174474,0.486611,1.513187,0.117716,-1.199281,-0.386473,-2.321946,-0.915143,-0.668993,-0.677160,-0.193064,0.744073,-1.521777,-1.040939,0.459815,-0.547519,-0.735954,-1.444746,0.361535,-0.222635,-0.516026,1.893558,2.091127,-1.892163,1.773132,0.139216,0.550950,0.488561,1.454481,0.693523,1.246853,0.417728,1.843159,1.748698,0.497601,-1.069408,-0.394639,0.815649,0.358375,0.326708,0.230385,0.354666,1.590914,-0.788364,0.647431,-0.444399,-0.175848,-0.002258,1.538451,1.063911,2.766802,0.671918,0.778368,-2.064291,1.015605,1.584417,1.134547,0.354193,0.661850,-2.307800,0.495789,0.680831,1.029797,2.171608,2.825816,-0.776524,-2.508737,0.023562,0.729947,-0.225823,-0.405032,-0.335337,-0.019626,0.171913,0.202361,1.037812,-2.352982,-0.261039,-0.930136,0.106995,-0.166257,-1.631306,-0.920781,-0.604181,-0.487815,-1.255186,2.044111,-0.119722,-0.226260]
# print(arr.__len__())
#
# zeros = np.zeros(5).reshape((1,5))
# ones = np.ones(5).reshape((1,5))
#
# print(zeros)
# print(ones)

# matrix = np.matrix([[0,0,0,0],[1,1,1,1],[1,1,1,1]])
# arr = [0,1,1]
#
# # np.concatenate(matrix, [0,0,0,0])
# matrix2 = np.empty([0,4], dtype=float)
# matrix2 = np.row_stack([matrix2,matrix[0]])
# matrix2 = np.row_stack([matrix2,matrix[1]])
# matrix2 = np.row_stack([matrix2,matrix[2]])
#
# print(matrix2)
#
# arr2 = []
# arr2 = np.append(arr2, 2)
#
# print(arr2)

# def balance_sampling(matrix, arr):
#     group = arr.group.count
#     print(group)
#
# balance_sampling(matrix, arr)
#
#
# for i in range(3):
#     print(arr[i])

# str = 'aaa \n bbb \n ccc'
# print(str)
# print(str.strip('\n'))
# print(str.replace('\n', ' '))

columns = ['location_traffic_convenience',
       'location_distance_from_business_district', 'location_easy_to_find',
       'service_wait_time', 'service_waiters_attitude',
       'service_parking_convenience', 'service_serving_speed', 'price_level',
       'price_cost_effective', 'price_discount', 'environment_decoration',
       'environment_noise', 'environment_space', 'environment_cleaness',
       'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
       'others_overall_experience', 'others_willing_to_consume_again']


file='/Users/sufeng/PycharmProjects/comments_sentiment_analysis/data/dict/test.txt'
df_validate = pd.DataFrame(pd.read_csv(file, header=0))
df_validate.set_index('id')

for i in range(2):
    for column in columns:
        df_validate.loc[i, column] = 0

for i in range(2):
    for column in columns:
        print(df_validate.loc[i, column])

a = np.array([[10, 2.7, 3.6],

              [-100, 5, -2],

              [120, 20, 40]],

             dtype=np.float64)

print(a)  # 比较在预处理前的数据

print(scale(a))  # 比较在预处理后的数据

b = np.array([[10, 2.7, 3.6]])

print(scale(b))



