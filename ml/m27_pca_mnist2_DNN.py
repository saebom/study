# 실습
# 아까 4가지 모델을 맹그러봐
# 784개 DNN으로 만든거 (최상의 성능인거 // 0.996 이상)과 비교

#1. 나의 최고의 DNN
# time = 59.989062547683716
# acc =  0.9761000275611877

#2. 나의 최고의 CNN
# time = 83.67315149307251
# acc = 0.9915000200271606

#3. PCA 0.95(XGBClassifier())
# time = 64.96638822555542
# acc =  0.9743

#4. PCA 0.99(RandomForestClassifier(), XGBClassifier())
# time = 58.80393075942993, 14.459177494049072
# acc = 0.9726, 0.9614

#5. PCA 0.999(RandomForestClassifier(), SVC(), XGBClassifier())
# time = 63.52199983596802, 174.18833303451538, 18.670639276504517
# acc = 0.9717,  0.9791,  0.9628

#6. PCA 1.0(RandomForestClassifier(), SVC(), XGBClassifier())
# time = 65.26432061195374,  175.72441720962524, 24.71903395652771
# acc = 0.9729,  0.9791,  0.9603

#====================================================================================================#
from tabnanny import verbose
from keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)  
x_test = x_test.reshape(10000, 784)   
# print(np.unique(x_train, return_counts=True))

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#PCA 주성분분석
from sklearn.decomposition import PCA
pca = PCA(n_components=331)   
                            
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
# print(x_train.shape, x_test.shape)  # (60000, 331) (10000, 331)

# pca_EVR = pca.explained_variance_ratio_
# cumsum = np.cumsum(pca_EVR) 

# print(np.argmax(cumsum >= 0.95)+1)    # 154
# print(np.argmax(cumsum >= 0.99)+1)    # 331
# print(np.argmax(cumsum >= 0.999)+1)   # 486
# print(np.argmax(cumsum >= 1.0)+1)     # 713

# One Hot Encoding
# import pandas as pd
# # df = pd.DataFrame(y)
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
# print(y_train)
# print(y_train.shape)


#2. 모델구성
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# model = SVC(verbose=1)
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)


#3. 훈련
import datetime
start_time = time.time()   
model.fit(x_train, y_train)
end_time = time.time() - start_time

#4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 : ', result)
print('걸린 시간 : ', end_time)


