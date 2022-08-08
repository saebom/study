import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold
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
# model = make_pipeline(MinMaxScaler(), RandomForestRegressor())    # pipeline에는 scaling과 model을 정의하지 않고 사용
pipe = Pipeline([('standard', StandardScaler()), ('RF', RandomForestRegressor())], verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

model = HalvingRandomSearchCV(pipe, parameters, cv=5, verbose=1)


#3. 훈련
model.fit(x_train, y_train) # make_pipeline에서의 fit은 fit_transform이 적용됨


#4. 평가, 예측
result = model.score(x_test, y_test)    # make_pipeline에서의 model.score는 transform이 적용됨
print('model.score : ', result) # model.score :  1.0



#=========================== pipeline & gridSearch 적용결과 =============================#
# model.score :  0.8059216694486484
#============================= HalvingRandomSearchCV 결과 ===============================#
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# 최적의 매개변수 :  RandomForestRegressor(min_samples_split=10, n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'min_samples_split': 10}
# best_score_ :  0.8374481265619045
# model.score :  0.9119201711683705
# r2_score :  0.9119201711683705
# 최적의 튠 ACC :  0.9119201711683705
# 걸린시간 :  8.97 초
#============================= HalvingGridSearchCV 결과 ===============================#
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# 최적의 매개변수 :  RandomForestRegressor(n_jobs=2)
# 최적의 파라미터 :  {'n_estimators': 100, 'n_jobs': 2}
# best_score_ :  0.8477590645172572
# model.score :  0.8889409837902635
# r2_score :  0.8889409837902635
# 최적의 튠 ACC :  0.8889409837902635
# 걸린시간 :  14.24 초
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

