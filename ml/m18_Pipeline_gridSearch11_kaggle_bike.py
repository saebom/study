import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')


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

from sklearn.model_selection import train_test_split, KFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=72
)
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)   # train은 fit_transform, test는 transform으로 overfit(과적합)이 안 잡힘

parameters = [
    {'RF__n_estimators' : [100, 200], 'RF__max_depth':[6, 8, 10, 12], 'RF__n_jobs' : [-1, 2, 4]},  #n_estimators 는 epoch
    {'RF__max_depth' : [6, 8, 10, 12], 'RF__min_samples_split' : [2, 3, 5, 10]},
    {'RF__n_estimators' : [100, 200], 'RF__min_samples_leaf' : [3, 5, 7, 10]},
    {'RF__min_samples_split' : [2, 3, 5, 10], 'RF__n_jobs' : [-1, 2, 4]}, 
    {'RF__n_estimators' : [100, 200],'RF__n_jobs' : [-1, 2, 4]}
]

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


#2. 모델
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline

# model = SVC()
# model = make_pipeline(StandardScaler(), SVR())    # pipeline에는 scaling과 model을 정의하지 않고 사용
# model = make_pipeline(StandardScaler(), RandomForestRegressor())    # pipeline에는 scaling과 model을 정의하지 않고 사용
# model = make_pipeline(MinMaxScaler(), RandomForestRegressor())    # pipeline에는 scaling과 model을 정의하지 않고 사용
pipe = Pipeline([('standard', StandardScaler()), ('RF', RandomForestRegressor())], verbose=1)


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

model = GridSearchCV(pipe, parameters, cv=5, verbose=1)


#3. 훈련
model.fit(x_train, y_train) # make_pipeline에서의 fit은 fit_transform이 적용됨


#4. 평가, 예측
result = model.score(x_test, y_test)    # make_pipeline에서의 model.score는 transform이 적용됨
print('model.score : ', result) # model.score :  1.0



#=========================== pipeline & gridSearch 적용결과 =============================#
# model.score :  0.8776339862098779
#============================= HalvingRandomSearchCV 결과 ===============================#
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'n_estimators': 100, 'max_depth': 10}
# best_score_ :  0.6442139826553579
# model.score :  0.8441955255894151
# r2_score :  0.8441955255894151
# 최적의 튠 ACC :  0.8441955255894151
# 걸린시간 :  17.32 초
#============================= HalvingRandomSearchCV 결과 ===============================#
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'n_estimators': 100, 'max_depth': 10}
# best_score_ :  0.6442139826553579
# model.score :  0.8441955255894151
# r2_score :  0.8441955255894151
# 최적의 튠 ACC :  0.8441955255894151
# 걸린시간 :  17.32 초
#============================= HalvingGridSearchCV 결과 ===============================#
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=12, n_estimators=200, n_jobs=4)
# 최적의 파라미터 :  {'max_depth': 12, 'n_estimators': 200, 'n_jobs': 4}
# best_score_ :  0.8548307690419179
# model.score :  0.8547555830206046
# r2_score :  0.8547555830206046
# 최적의 튠 ACC :  0.8547555830206046
# 걸린시간 :  31.6 초
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

