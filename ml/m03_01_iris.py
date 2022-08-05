import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.random.set_seed(66)  # weight에 난수값 

#1. 데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)   # 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'

x = datasets['data']
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape)  # (150, 4), (150,)
print('y의 라벨값 : ', np.unique(y))    # y의 라벨값 :  [0 1 2]

x_train, x_test, y_train, y_test = train_test_split(
   x, y, train_size=0.8, shuffle=True, random_state=72
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression     #LogisticRegression은 분류모델
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = LinearSVC()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()


#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
results = model.score(x_test, y_test)                                         
print('결과 acc : ', results)



#===================================== 결  과 ==========================================#
# LinearSVC() 결과 acc : 1.0
# LogisticRegression() 결과 acc :  1.0
# KNeighborsClassifier() 결과 acc :  0.9666666666666667
# DecisionTreeClassifier() 결과 acc :  1.0
# RandomForestClassifier() 결과 acc :  1.0
#=======================================================================================#

