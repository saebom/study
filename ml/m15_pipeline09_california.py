import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72
    )
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)   # train은 fit_transform, test는 transform으로 overfit(과적합)이 안 잡힘


#2. 모델
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

# model = SVC()
# model = make_pipeline(StandardScaler(), SVR())    # pipeline에는 scaling과 model을 정의하지 않고 사용
# model = make_pipeline(StandardScaler(), RandomForestRegressor())    # pipeline에는 scaling과 model을 정의하지 않고 사용
model = make_pipeline(MinMaxScaler(), RandomForestRegressor())    # pipeline에는 scaling과 model을 정의하지 않고 사용


#3. 훈련
model.fit(x_train, y_train) # make_pipeline에서의 fit은 fit_transform이 적용됨


#4. 평가, 예측
result = model.score(x_test, y_test)    # make_pipeline에서의 model.score는 transform이 적용됨
print('model.score : ', result) # model.score :  1.0



#==================================== pipeline 적용결과 =================================#
# model.score :  0.8127376870341386
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
