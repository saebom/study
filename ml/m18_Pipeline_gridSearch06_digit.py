import numpy as np
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72
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
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)


#2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline

# model = SVC()
# model = make_pipeline(MinMaxScaler(), SVC())    # pipeline에는 scaling과 model을 정의하지 않고 사용
# model = make_pipeline(MinMaxScaler(), RandomForestClassifier())    # pipeline에는 scaling과 model을 정의하지 않고 사용
# model = make_pipeline(StandardScaler(), RandomForestClassifier())    # pipeline에는 scaling과 model을 정의하지 않고 사용
pipe = Pipeline([('standard', StandardScaler()), ('RF', RandomForestClassifier())], verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

model = GridSearchCV(pipe, parameters, cv=5, verbose=1)

#3. 훈련
model.fit(x_train, y_train) # make_pipeline에서의 fit은 fit_transform이 적용됨


#4. 평가, 예측
result = model.score(x_test, y_test)    # make_pipeline에서의 model.score는 transform이 적용됨
print('model.score : ', result) 


#=========================== pipeline & gridSearch 적용결과 =============================#
# model.score :  0.9805555555555555
#============================= HalvingRandomSearchCV 결과 ===============================#
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# 최적의 매개변수 :  RandomForestClassifier(n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'min_samples_split': 2}
# best_score_ :  0.9621104903786467
# model.score :  0.9777777777777777
# accuracy_score :  0.9777777777777777
# 최적의 튠 ACC :  0.9777777777777777
# 걸린시간 :  5.11 초
#============================= HalvingGridSearchCV 결과 ===============================#       
# Fitting 5 folds for each of 16 candidates, totalling 80 fits
# 최적의 매개변수 :  RandomForestClassifier(max_depth=12, n_jobs=4)
# 최적의 파라미터 :  {'max_depth': 12, 'min_samples_split': 2, 'n_estimators': 100, 'n_jobs': 4}
# best_score_ :  0.9643513345747982
# model.score :  0.9740740740740741
# accuracy_score :  0.9740740740740741
# 최적의 튠 ACC :  0.9740740740740741
# 걸린시간 :  39.38 초
#============================== RandomizedSearchCV 결과 ===============================#
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestClassifier(max_depth=12, min_samples_split=3, n_estimators=200,
#                        n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'n_estimators': 200, 'min_samples_split': 3, 'max_depth': 12}
# best_score_ :  0.9681907291469045
# model.score :  0.9814814814814815
# accuracy_score :  0.9814814814814815
# 최적의 튠 ACC :  0.9814814814814815
# 걸린시간 :  4.77 초
#================================= GridSearchCV 결과 ===================================#
# Fitting 5 folds for each of 138 candidates, totalling 690 fits
# 최적의 매개변수 :  RandomForestClassifier(n_jobs=-1)
# 최적의 파라미터 :  {'min_samples_split': 2, 'n_jobs': -1}
# best_score_ :  0.9737557705685196
# model.score :  0.9833333333333333
# accuracy_score :  0.9833333333333333
# 최적의 튠 ACC :  0.9833333333333333
# 걸린시간 :  33.86 초
# =======================================================================================#

