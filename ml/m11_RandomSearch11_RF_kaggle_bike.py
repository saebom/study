import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import r2_score


#1. 데이터
path = './_data/bike/'
train_set = pd.read_csv(path + 'train.csv')
print(train_set)
print(train_set.shape)   # (10886, 11)
print(train_set.columns) # ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                         #  'humidity', 'windspeed', 'casual', 'registered', 'count']
                        
print(train_set.info())
print(train_set.describe())
print(train_set.isnull().sum())

test_set = pd.read_csv(path + 'test.csv')
print(test_set)
print(test_set.shape)   # (6493, 8)
print(test_set.columns) # ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                        #  'humidity', 'windspeed']   ==> casual, registered, count
print(test_set.info())
print(test_set.describe())
print(test_set.isnull().sum())  

#### 결측치 처리 ####
train_set = train_set[train_set['weather'] != 4]  # weather가 4인 데이터는 이상치(폭우, 폭설이 내리는 날 저녁 6시에 대여)

from datetime import datetime
train_set['datetime'] = pd.to_datetime(train_set['datetime'])
test_set['datetime'] = pd.to_datetime(test_set['datetime'])

# 날짜 feature 생성

L = ['year', 'month', 'date', 'hour', 'weekday']
train_set = train_set.join(pd.concat([getattr(train_set['datetime'].dt, i).rename(i) for i in L], axis=1))
test_set = test_set.join(pd.concat([getattr(test_set['datetime'].dt, i).rename(i) for i in L], axis=1))

print(train_set)
print(test_set)

# drop_features

drop_features = ['casual', 'registered', 'datetime', 'year', 'date', 'windspeed', 'month']
train_set = train_set.drop(drop_features, axis=1)
test_set = test_set.drop(['datetime', 'year', 'date', 'windspeed', 'month'], axis=1)

print("train_set.columns :", train_set.columns)
x = train_set.drop(['count'], axis=1)
print(x)
print(x.columns)
# print(x.shape)  

y = train_set['count']
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=13
)

n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1004)

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
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Perceptron, LogisticRegression     #LogisticRegression은 분류모델
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# model = RandomForestRegressor(max_depth=100, n_estimators=100, n_jobs=2) 
# model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1,
#                      refit=True, n_jobs=-1)
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1,
                     refit=True, n_jobs=-1)

#3. 컴파일, 훈련
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
# 최적의 매개변수 :  RandomForestRegressor(min_samples_split=3, n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'min_samples_split': 3}
# best_score_ :  0.8556260815591286
# model.score :  0.8582739946801704
# r2_score :  0.8582739946801704
# 최적의 튠 ACC :  0.8582739946801704
# 걸린시간 :  13.25 초
#================================= GridSearchCV 결과 ===================================#
# Fitting 5 folds for each of 66 candidates, totalling 330 fits
# 최적의 매개변수 :  RandomForestRegressor(min_samples_split=5, n_jobs=2)
# 최적의 파라미터 :  {'min_samples_split': 5, 'n_jobs': 2}
# best_score_ :  0.8565500965640249
# model.score :  0.8616759505288746
# r2_score :  0.8616759505288746
# 최적의 튠 ACC :  0.8616759505288746
# 걸린시간 :  68.08 초
# =======================================================================================#

