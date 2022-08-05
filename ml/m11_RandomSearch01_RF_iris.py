# 실습
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV

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


'''모델 : RandomForestClassifier의 파라미터 ========================================================

parameters = [
    {'n_estimators' : [100, 200]},  #n_estimators 는 epoch
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10]}, 
    {'n_jobs' : [-1, 2, 4]}
]

# 파라미터 조합으로 2개 이상 엮을 것
# ================================================================================================'''
parameters = [
    {'n_estimators' : [100, 200], 'max_depth':[6, 8, 10, 12], 'n_jobs' : [-1, 2, 4]},  #n_estimators 는 epoch
    {'max_depth' : [6, 8, 10, 12], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_estimators' : [100, 200], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4]}, 
    {'n_estimators' : [100, 200],'n_jobs' : [-1, 2, 4]}
]

#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression     #LogisticRegression은 분류모델
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier(n_estimators=100203, max_depth=6, n_jobs=2) 
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, 
                     refit=True, n_jobs=-1)                  


#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train) 
print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print("best_score_ : ", model.best_score_)
print("model.score : ", model.score(x_test, y_test))
end_time = time.time() 

#4. 평가, 예측
y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적의 튠 ACC : ', accuracy_score(y_test, y_pred_best))
print('걸린시간 : ', round(end_time-start, 2), "초")

#============================== RandomizedSearchCV 결과 ===============================#
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=10)  
# 최적의 파라미터 :  {'n_estimators': 100, 'min_samples_leaf': 10}
# best_score_ :  0.9666666666666668
# model.score :  1.0
# accuracy_score :  1.0
# 최적의 튠 ACC :  1.0
# 걸린시간 :  3.3 초 
#================================= GridSearchCV 결과 ===================================#
# Fitting 5 folds for each of 66 candidates, totalling 330 fits
# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, n_jobs=-1)     
# 최적의 파라미터 :  {'max_depth': 6, 'n_estimators': 100, 'n_jobs': -1}
# best_score_ :  0.95
# model.score :  1.0
# accuracy_score :  1.0
# 최적의 튠 ACC :  1.0
# 걸린시간 :  10.61 초
# =======================================================================================#

