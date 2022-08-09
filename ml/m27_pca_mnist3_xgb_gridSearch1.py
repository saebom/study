# n_component > 0.95 이상
# xgboost, gridSearch 또는 RandomSearch를 쓸 것

# m27_2 결과를 뛰어넘어랏!!

# parameters = [
#     {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01],
#      'max_depth':[4,5,6]},
#     {'n_estimators':[90, 100, 110], 'learning_rate':[0.1, 0.001, 0.01],
#      'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1]},
#     {'n_estimators':[90, 110], 'learning_rate':[0.1, 0.001, 0.5],
#      'max_depth':[4,5,6], 'colsample_bytree':[0.6, 0.9, 1],
#      'colsample_bylevel':[0.6,0.7.0.9]},
# ]
# n_jobs = -1
# tree_mothod = 'gpu_hist', 
# predictor = 'gpu_predictor',
# gpu_id = 0

# 실습 시작!!

from keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)  
x_test = x_test.reshape(10000, 784)   
# print(np.unique(x_train, return_counts=True))

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#PCA 주성분분석
from sklearn.decomposition import PCA
pca = PCA(n_components=154)   
                            
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
# print(x_train.shape, x_test.shape)  # (60000, 331) (10000, 331)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=72)

parameters = [
    {'n_estimators':[100, 200], 'learning_rate':[0.1, 0.3, 0.001],
     'max_depth':[4,5]},
    {'n_estimators':[90, 100], 'learning_rate':[0.1, 0.01],
     'max_depth':[5,6], 'colsample_bytree':[0.6, 1]},
    {'n_estimators':[90, 110], 'learning_rate':[0.1, 0.5],     
     'max_depth':[4,6], 'colsample_bytree':[0.6, 1], 'colsample_bylevel':[0.6,0.9]},
]


#2. 모델구성
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# model = SVC(verbose=1)
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
model = RandomizedSearchCV(XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0), 
                     parameters, cv=kfold, verbose=2, 
                     refit=True, n_jobs=-1)                  


# 3. 훈련
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


#==================================== 결과 ==================================#
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 파라미터 :  {'n_estimators': 110, 'max_depth': 6, 'learning_rate': 0.5, 'colsample_bytree': 0.6, 'colsample_bylevel': 0.6}
# best_score_ :  0.9583
# model.score :  0.9641
# accuracy_score :  0.9641
# 최적의 튠 ACC :  0.9641
# 걸린시간 :  740.81 초

# 최적의 파라미터 :  {'n_estimators': 90, 'max_depth': 6, 'learning_rate': 0.5, 'colsample_bytree': 1, 'colsample_bylevel': 0.6}
# best_score_ :  0.9593666666666667
# model.score :  0.9627
# accuracy_score :  0.9627
# 최적의 튠 ACC :  0.9627
# 걸린시간 :  572.93 초
#============================================================================#
