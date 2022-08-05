from weakref import ref
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVR, SVR


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=66
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
# best_score_ :  0.8479916565486251
# model.score :  0.8865442848992637
# r2_score :  0.8865442848992637
# 최적의 튠 ACC :  0.8865442848992637
# 걸린시간 :  3.77 초
#================================= GridSearchCV 결과 ===================================#
# Fitting 5 folds for each of 66 candidates, totalling 330 fits
# 최적의 매개변수 :  RandomForestRegressor(min_samples_split=3, n_jobs=-1)
# 최적의 파라미터 :  {'min_samples_split': 3, 'n_jobs': -1}
# best_score_ :  0.8475395356400337
# model.score :  0.8925075847843841
# r2_score :  0.8925075847843841
# 최적의 튠 ACC :  0.8925075847843841
# 걸린시간 :  12.3 초
# =======================================================================================#

