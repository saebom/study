import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
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
# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, 
#                      refit=True, n_jobs=-1)                  
# model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, 
#                      refit=True, n_jobs=-1)
# model = HalvingGridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, 
#                      refit=True, n_jobs=-1)
model = HalvingRandomSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, 
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
        

#============================= HalvingRandomSearchCV 결과 ===============================#
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=10, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 10}
# best_score_ :  0.961111111111111
# model.score :  0.9649122807017544
# accuracy_score :  0.9649122807017544
# 최적의 튠 ACC :  0.9649122807017544
# 걸린시간 :  5.86 초
#============================= HalvingGridSearchCV 결과 ===============================#
# Fitting 5 folds for each of 22 candidates, totalling 110 fits
# 최적의 매개변수 :  RandomForestClassifier(n_jobs=4)
# 최적의 파라미터 :  {'min_samples_split': 2, 'n_jobs': 4}
# best_score_ :  0.9444444444444443
# model.score :  1.0
# accuracy_score :  1.0
# 최적의 튠 ACC :  1.0
# 걸린시간 :  13.09 초
#============================== RandomizedSearchCV 결과 ===============================#
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 3}
# best_score_ :  0.9673101265822786
# model.score :  0.9707602339181286
# accuracy_score :  0.9707602339181286
# 최적의 튠 ACC :  0.9707602339181286
# 걸린시간 :  3.73 초
#================================= GridSearchCV 결과 ===================================#
# Fitting 5 folds for each of 66 candidates, totalling 330 fits
# 최적의 매개변수 :  RandomForestClassifier(n_jobs=4)
# 최적의 파라미터 :  {'min_samples_split': 2, 'n_jobs': 4}
# best_score_ :  0.969873417721519
# model.score :  0.9590643274853801
# accuracy_score :  0.9590643274853801
# 최적의 튠 ACC :  0.9590643274853801
# 걸린시간 :  10.85 초
# =======================================================================================#
