import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = fetch_covtype()
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
    # {'RF__max_depth' : [6, 8, 10, 12], 'RF__min_samples_split' : [2, 3, 5, 10]},
    {'RF__n_estimators' : [100, 200], 'RF__min_samples_leaf' : [3, 5, 7, 10]},
    # {'RF__min_samples_split' : [2, 3, 5, 10], 'RF__n_jobs' : [-1, 2, 4]}, 
    # {'RF__n_estimators' : [100, 200],'RF__n_jobs' : [-1, 2, 4]}
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
print('model.score : ', result) # model.score :  1.0


#=========================== pipeline & gridSearch 적용결과 =============================#
# model.score :  0.9458705885390223
#============================= HalvingRandomSearchCV 결과 ===============================#
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# 최적의 매개변수 :  RandomForestClassifier(n_estimators=200, n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'n_estimators': 200}
# best_score_ :  0.771957671957672
# model.score :  0.9582732513310078
# accuracy_score :  0.9582732513310078
# 최적의 튠 ACC :  0.9582732513310078
# 걸린시간 :  150.04 초
#============================= HalvingGridSearchCV 결과 ===============================#
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# 최적의 매개변수 :  RandomForestClassifier(n_estimators=200, n_jobs=4)
# 최적의 파라미터 :  {'n_jobs': 4, 'n_estimators': 200}
# best_score_ :  0.9478015630884826
# model.score :  0.9536442078208188
# accuracy_score :  0.9536442078208188
# 최적의 튠 ACC :  0.9536442078208188
# 걸린시간 :  821.33 초
#============================== RandomizedSearchCV 결과 ===============================#
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=5, n_estimators=200)
# 최적의 파라미터 :  {'n_estimators': 200, 'min_samples_leaf': 5}
# best_score_ :  0.9241814744697446
# model.score :  0.9319636956122637
# accuracy_score :  0.9319636956122637
# 최적의 튠 ACC :  0.9319636956122637
# 걸린시간 :  856.85 초
#================================= GridSearchCV 결과 ===================================#
# Fitting 5 folds for each of 138 candidates, totalling 690 fits
# 최적의 매개변수 :  RandomForestClassifier(n_estimators=200, n_jobs=2)
# 최적의 파라미터 :  {'n_estimators': 200, 'n_jobs': 2}
# best_score_ :  0.947761047248018
# model.score :  0.9533401413622178
# accuracy_score :  0.9533401413622178
# 최적의 튠 ACC :  0.9533401413622178
# 걸린시간 :  9050.24 초
# =======================================================================================#