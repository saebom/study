import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv',   # 예측에서 사용!!
                       index_col=0)

#### 결측치 처리 ####
test_set = test_set.fillna(method='ffill')
train_set = train_set.dropna()  # nan 값 삭제

x = train_set.drop(['count'], axis=1)
y = train_set['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, random_state=72
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
# model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1,
#                      refit=True, n_jobs=-1)
# model = HalvingGridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1,
#                      refit=True, n_jobs=-1)
model = HalvingRandomSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1,
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


#============================= HalvingRandomSearchCV 결과 ===============================#
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, n_jobs=4)
# 최적의 파라미터 :  {'n_jobs': 4, 'n_estimators': 100, 'max_depth': 10}
# best_score_ :  0.710188776958297
# model.score :  0.7400396579575074
# r2_score :  0.7400396579575075
# 최적의 튠 ACC :  0.7400396579575074
# 걸린시간 :  15.16 초
#============================= HalvingGridSearchCV 결과 ===============================#
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, n_jobs=2)
# 최적의 파라미터 :  {'max_depth': 10, 'n_estimators': 100, 'n_jobs': 2}
# best_score_ :  0.7589764284695842
# model.score :  0.7421211581161351
# r2_score :  0.7421211581161351
# 최적의 튠 ACC :  0.7
#============================== RandomizedSearchCV 결과 ===============================#
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, n_estimators=200, n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'n_estimators': 200, 'max_depth': 10}
# best_score_ :  0.7581431300185069
# model.score :  0.7497690244094714
# r2_score :  0.7497690244094714
# 최적의 튠 ACC :  0.7497690244094714
# 걸린시간 :  5.0 초
#================================= GridSearchCV 결과 ===================================#
# Fitting 5 folds for each of 66 candidates, totalling 330 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=12, min_samples_split=3)
# 최적의 파라미터 :  {'max_depth': 12, 'min_samples_split': 3}
# best_score_ :  0.7615821162385703
# model.score :  0.7507763823829572
# r2_score :  0.7507763823829572
# 최적의 튠 ACC :  0.7507763823829572
# 걸린시간 :  16.75 초
# =======================================================================================#


