import numpy as np
from sklearn import datasets
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.95, random_state=72
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
# 최적의 매개변수 :  RandomForestRegressor(max_depth=12, n_estimators=200, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'n_estimators': 200, 'max_depth': 12}
# best_score_ :  0.6330669843433865
# model.score :  0.8167192395088646
# r2_score :  0.8167192395088647
# 최적의 튠 ACC :  0.8167192395088646
# 걸린시간 :  18.25 초
#============================= HalvingGridSearchCV 결과 ===============================#
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestRegressor(n_estimators=200, n_jobs=2)
# 최적의 파라미터 :  {'n_estimators': 200, 'n_jobs': 2}
# best_score_ :  0.8021646635801369
# model.score :  0.814893386956294
# r2_score :  0.814893386956294
# 최적의 튠 ACC :  0.814893386956294
# 걸린시간 :  72.04 초
#============================== RandomizedSearchCV 결과 ===============================#
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestRegressor(n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'min_samples_split': 2}
# best_score_ :  0.8023923216832938
# model.score :  0.8148554665360241
# r2_score :  0.8148554665360241
# 최적의 튠 ACC :  0.8148554665360241
# 걸린시간 :  53.49 초
#================================= GridSearchCV 결과 ===================================#
# Fitting 5 folds for each of 66 candidates, totalling 330 fits
# 최적의 매개변수 :  RandomForestRegressor(n_estimators=200, n_jobs=4)
# 최적의 파라미터 :  {'n_estimators': 200, 'n_jobs': 4}
# best_score_ :  0.802793091487606
# model.score :  0.8145915644407198
# r2_score :  0.8145915644407198
# 최적의 튠 ACC :  0.8145915644407198
# 걸린시간 :  284.8 초
# =======================================================================================#


