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
    x, y, shuffle=True, random_state=123, train_size=0.8
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = {'n_estimators': [100],
              'learning_rate' : [0.1, 0.2],
              'max_depth' : [3,4,5], #default 6 => 통상 max는 4정도에서 성능이 좋다
              'gamma': [1,2],
              'min_child_weight': [1,5],
              'subsample' : [0.7,1],
              'colsample_bytree' : [0.7,1],
              'colsample_bylevel' : [0.7,1],
              'colsample_bynode' : [0.7,1],
              'reg_alpha' : [0, 0.1],
              'reg_lambda' : [0, 0.1],
              }  


#2. 모델
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV

xgb = XGBRegressor(random_state=123,
                    )
model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)    
print('최상의 매개변수 : ', model.best_params_)
print('최상의 점수 : ', model.best_score_)
print('acc : ', result)

#=================================== 결과 =====================================#
# 최상의 매개변수 :  {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 
# 'gamma': 1, 'learning_rate': 0.2, 'max_depth': 5, 'min_child_weight': 5, 
# 'n_estimators': 100, 'reg_alpha': 0, 'reg_lambda': 0, 'subsample': 1}
# 최상의 점수 :  0.8288049509136419
# acc :  0.8364322247332923
#==============================================================================#
