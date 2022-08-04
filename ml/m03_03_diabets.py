import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.svm import LinearSVR

import tensorflow as tf
tf.random.set_seed(777)

#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=72
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (309, 10) (133, 10) (309,) (133,)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression, LinearRegression     #LogisticRegression은 분류모델, LinearRegression 회귀모델
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# model = LinearSVR()
# model = LinearRegression()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
results = model.score(x_test, y_test)
print('결과 r2 : ', results)



#===================================== 결  과 ==========================================#
# LinearSVR() 결과 r2 :  0.16041096520251164
# LinearRegression() 결과 r2 :  0.6307176518001982
# KNeighborsRegressor() 결과 r2 :  0.5422056859784175
# DecisionTreeRegressor() 결과 결과 r2 :  -0.0299860886516623
# RandomForestRegressor() 결과 r2 : 0.5696018108891183
#=======================================================================================#


