# csv로 맹그러!!!

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel

#1. 데이터
path = 'D:/study_data/_data/'
data_set = pd.read_csv(path + 'winequality-white.csv', index_col=None, header=0, sep=';')

print('data_set.shape', data_set.shape)    # (4898, 12)
print(data_set.describe())
print(data_set.info())

import matplotlib.pyplot as plt
######################## 그래프 그려봐 !!! ############################
# 1. value_counts -> 쓰지마!!!
# 2. groupby 써, count() 써!!!

# plt.bar 로 그린다. (quality 컬럼)

count_data = data_set.groupby('quality')['quality'].count()
print(count_data)
plt.bar(count_data.index, count_data)
plt.show()

'''
# x = data_set.drop(['quality'], axis=1)
# y = data_set['quality']

data_set2 = data_set.to_numpy()
# data_set2 = data_set.values
print(type(data_set2))
print(data_set2.shape)  # (4898, 12)

x = data_set2[:, :11]
y = data_set2[:, 11]
print(x.shape, y.shape) # (4898, 11) (4898,)
print(np.unique(y, return_counts=True))  # (array([3., 4., 5., 6., 7., 8., 9.]), 
                                         # array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
print(data_set['quality'].value_counts)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, shuffle=True, train_size=0.8,
    stratify=y
)

#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
from sklearn.metrics import accuracy_score, f1_score

y_predict = model.predict(x_test) 
score = model.score(x_test, y_test)
print('model.score : ', score)     # model.score : 0.7224489795918367
print('acc_score : ', accuracy_score(y_test, y_predict))
print('f1_score(macro) : ', f1_score(y_test, y_predict, average='macro'))   # f1_score는 이진분류이므로 average를 사용하여 다중분류에 사용
print('f1_score(micro) : ', f1_score(y_test, y_predict, average='micro'))   #
'''

 
#================================== 결 과 =====================================#
# model.score :  0.7316326530612245
# acc_score :  0.7316326530612245
# f1_score(macro) :  0.4436222951519518
# f1_score(micro) :  0.7316326530612245
#==============================================================================#
