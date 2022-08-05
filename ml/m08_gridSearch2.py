import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold

import tensorflow as tf
tf.random.set_seed(66)  # weight에 난수값 

#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)   # 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'

x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
   x, y, train_size=0.8, shuffle=True, random_state=72
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

# SVC 모델의 parmeters
parameters =[
   {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3, 4, 5]},    # 12번
   {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},        # 6번
   {"C":[1, 10, 100, 1000], "kernel":["sigmoid"],                        # 24번
    "gamma":[0.01, 0.001, 0.0001], "degree":[3, 4]}                      
]                                                                        # 총 42번


#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression     #LogisticRegression은 분류모델
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = SVC(C=1, kernel='linear', degree=3) # 33번 라인 parameters 12회
# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1, # GridSearchCV의 본래 목적은 최적 하이퍼 파라미터를 찾는 것. refit을 False로 하면 최적 하이퍼 파라미터만 찾아준
                    #  refit=True, n_jobs=-1)                  # n_jobs 는 cpu 갯수, -1면 전체 사용


#3. 훈련
model.fit(x_train, y_train) 
# print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear')

# print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

# print("best_score_ : ", model.best_score_)
# best_score_ :  0.9583333333333334

print("model.score : ", model.score(x_test, y_test))
# model.score ;  1.0


#4. 평가, 예측
y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))
# accuracy_score :  1.0

# y_pred_best = model.best_estimator_.predict(x_test)
# print('최적의 튠 ACC : ', accuracy_score(y_test, y_pred_best))

# 최적의 튠 ACC :  1.0

#===================================== 결  과 ==========================================#
# model.score ;  1.0
# accuracy_score :  1.0
#=======================================================================================#


