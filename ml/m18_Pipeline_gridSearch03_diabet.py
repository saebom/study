import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_diabetes()
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
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


#2. 모델
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline

# model = SVC()
# model = make_pipeline(StandardScaler(), SVR())    # pipeline에는 scaling과 model을 정의하지 않고 사용
# model = make_pipeline(StandardScaler(), RandomForestRegressor())    # pipeline에는 scaling과 model을 정의하지 않고 사용
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

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)
print('r2_score : ', acc)


#=========================== pipeline & gridSearch 적용결과 =============================#
# model.score :  0.5846780582908867
# r2_score :  0.5846780582908867
#============================= HalvingRandomSearchCV 결과 ===============================#
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=8, n_estimators=200, n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'n_estimators': 200, 'min_samples_split': 2, 'max_depth': 8}
# best_score_ :  0.43226496438035455
# model.score :  0.5708865831750471
# r2_score :  0.5708865831750471
# 최적의 튠 ACC :  0.5708865831750471
# 걸린시간 :  9.01 초
#============================== RandomizedSearchCV 결과 ===============================#
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=8, min_samples_split=10, n_estimators=200,
#                       n_jobs=2)
# 최적의 파라미터 :  {'max_depth': 8, 'min_samples_split': 10, 'n_estimators': 200, 'n_jobs': 2}
# best_score_ :  0.4375209066196394
# model.score :  0.5716490447865309
# r2_score :  0.571649044786531
# 최적의 튠 ACC :  0.5716490447865309
# 걸린시간 :  32.2 초
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

