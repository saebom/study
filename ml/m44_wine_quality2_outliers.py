# 아웃라이어 확인
# 아웃라이어 처리
# 돌려봐

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

# 아웃라이어 확인
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,
                                               [25, 50, 75])
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr : ", iqr)    
    lower_bound = quartile_1 - (iqr *1.5)   
    upper_bound = quartile_3 + (iqr *1.5)   
    print("lower_bound : ", lower_bound)
    print("upper_bound : ", upper_bound)
    return np.where((data_out>upper_bound) |
                    (data_out<lower_bound))
    
outliers_loc= outliers(data_set2)
print("0열의 이상치의 위치 : ", outliers_loc)
    
import matplotlib.pyplot as plt
# plt.boxplot(data_set2)
# plt.show()  

# 아웃라이어 처리
import math
def outliers_printer(dataset):
    collist = []
    plt.figure(figsize=(10,8))
    for i in range(dataset.shape[1]):
        col = dataset[:, i]
        outliers_loc = outliers(col)
        print(i, '열의 이상치의 위치: ', outliers_loc, '\n')
        plt.subplot(math.ceil(dataset.shape[1]/2),2,i+1)
        plt.boxplot(col)
        plt.title(i)
        collist.append(i)
        collist.append(outliers_loc)
        
    plt.show()
    return collist

outwhere = outliers(x)
print(outwhere)
collist = outliers_printer(x)
# print(np.array(collist))

# x = np.delete(x, outliers_loc, 0)
# y = np.delete(y, outliers_loc, 0)


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
print('f1_score(micro) : ', f1_score(y_test, y_predict, average='micro'))   


