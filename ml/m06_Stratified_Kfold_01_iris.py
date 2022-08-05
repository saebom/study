import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold

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

# x_train, x_test, y_train, y_test = train_test_split(
#    x, y, train_size=0.8, shuffle=True, random_state=72
# )
n_splits=5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)   #StratifiedKFold는 y라벨이 일정한 비율만큼 잘리게 됨
                                                                            #그러나 KFold 자체에 분류하여 적용되고 있음


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
scores = cross_val_score(model, x, y, cv=kfold)
# scores = cross_val_score(model, x, y, cv=5)
print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))


#===================================== 결  과 ==========================================#
# ACC :  [0.96666667 0.96666667 1.         0.9        0.96666667] 
#  cross_val_score :  0.96
#=======================================================================================#


