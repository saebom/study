import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score


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
print(y_test)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression     #LogisticRegression은 분류모델
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = SVC()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()


#3.4. 컴파일, 훈련, 평가, 예측
# model.fit(x_train, y_train) 
scores = cross_val_score(model, x_train, y_train, cv=kfold)
# scores = cross_val_score(model, x, y, cv=5)
print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)  # [0 1 2 2 2 1 1 2 0 0 0 0 1 0 0 1 1 0 0 0 2 2 0 1 1 2 0 2 2 0]

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC : ', acc)


#===================================== 결  과 ==========================================#
# ACC :  [0.83333333 1.         1.         0.95833333 1.        ] 
#  cross_val_score :  0.9583
# cross_val_predict ACC :  0.9333333333333333
#=======================================================================================#


