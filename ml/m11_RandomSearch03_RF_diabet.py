import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
tf.random.set_seed(777)

#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=72
)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (309, 10) (133, 10) (309,) (133,)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


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
    {'n_estimators' : [100, 200], 'max_depth':[6, 8, 10, 12], 'min_samples_split':[2, 3, 5, 10], 
     'n_jobs' : [-1, 2, 4]},  #n_estimators 는 epoch
    {'max_depth' : [6, 8, 10, 12], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_estimators' : [100, 200], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4]}, 
    {'n_estimators' : [100, 200],'n_jobs' : [-1, 2, 4]}
]


#2. 모델 구성
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Perceptron, LogisticRegression     #LogisticRegression은 분류모델
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# model = RandomForestRegressor(max_depth=100, n_estimators=100, n_jobs=2) 
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1,
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
print('r2_score : ', r2_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적의 튠 ACC : ', r2_score(y_test, y_pred_best))
print('걸린시간 : ', round(end_time-start, 2), "초")
        

#============================== RandomizedSearchCV 결과 ===============================#
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=12, min_samples_split=10, n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'n_estimators': 100, 'min_samples_split': 10, 'max_depth': 12}
# best_score_ :  0.36403372243642185
# model.score :  0.5574813523735
# r2_score :  0.5574813523735003
# 최적의 튠 ACC :  0.5574813523735003
# 걸린시간 :  3.92 초
#================================= GridSearchCV 결과 ===================================#
# Fitting 5 folds for each of 138 candidates, totalling 690 fits
# 최적의 매개변수 :  RandomForestRegressor(min_samples_leaf=10, n_estimators=200)
# 최적의 파라미터 :  {'min_samples_leaf': 10, 'n_estimators': 200}
# best_score_ :  0.3837238266426172
# model.score :  0.5884135342268182
# r2_score :  0.5884135342268182
# 최적의 튠 ACC :  0.5884135342268182
# 걸린시간 :  24.7 초
# =======================================================================================#

