
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_diabetes
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time


#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target
print(x.shape, y.shape)     # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# bayesian_params = {
#     'max_depth' : [2,10], #default 3/ 0~inf(무한대) / 정수 => 소수점은 정수로 변환하여 적용해야 함
#     'gamma': [0,100], #default 0 / 0~inf
#     'min_child_weight': [0,100], #default 1 / 0~inf
#     'subsample' : [0.9,1], #default 1 / 0~1
#     'colsample_bytree' : [0,1], #default 1 / 0~1
#     'colsample_bylevel' : [0.8,1], #default 1 / 0~1
#     'colsample_bynode' : [0.9,1], #default 1 / 0~1
#     'reg_alpha' : [9,100], #default 0 / 0~inf / L1 절대값 가중치 규제 / 그냥 alpha도 적용됨
#     'reg_lambda' : [2,100], #default 1 / 0~inf / L2 제곱 가중치 규제 / 그냥 lambda도 적용됨
# }

# def lgb_hamsu(max_depth, gamma, min_child_weight,  
#               subsample, colsample_bytree, colsample_bylevel, colsample_bynode, 
#               reg_lambda, reg_alpha):
#     params = {
#         'n_estimators' : 500, 'learning_rate' : 0.02,
#         'max_depth' : int(round(max_depth)),                # 무조건 정수
#         'gamma' : int(round(gamma)), 
#         'min_child_weight' : int(round(min_child_weight)),  
#         'subsample' : max(min(subsample, 1), 0),             # 0~1 사이의 값
#         'colsample_bytree' : max(min(colsample_bytree, 1), 0),   
#         'colsample_bylevel' : max(min(colsample_bylevel, 1), 0),   
#         'colsample_bynode' : max(min(colsample_bynode, 1), 0),   
#         'reg_lambda' : max(reg_lambda, 0),          # 무조건 양수만
#         'reg_alpha' : max(reg_alpha, 0),        
#     }
    
    
#     # *여러개의인자를받겠다
#     # **키워드를 받겠다(딕셔너리 형태)
#     model = LGBMRegressor(**params)
    
#     model.fit(x_train, y_train,
#               eval_set=[(x_train, y_train), (x_test, y_test)],
#               eval_metric='rmse',
#               verbose=0,
#               early_stopping_rounds=50
#               )
    
#     y_predict = model.predict(x_test)
#     results = r2_score(y_test, y_predict)
    
#     return results

# lgb_bo = BayesianOptimization(f=lgb_hamsu, 
#                               pbounds=bayesian_params,
#                               random_state=123,
#                               )
# lgb_bo.maximize(init_points=5,
#                 n_iter=50)
# print(lgb_bo.max)
# {'target': 0.6162139000753267, 'params': {'colsample_bylevel': 0.8558216995107935, 
# 'colsample_bynode': 0.9839457783040828, 'colsample_bytree': 0.566945737927881, 
# 'gamma': 45.51493959236752, 'max_depth': 8.17108048422443, 
# 'min_child_weight': 27.93478193523481, 'reg_alpha': 17.411229495773064, 
# 'reg_lambda': 26.770817561390476, 'subsample': 0.946050112624914}}


###################################### [실습] ########################################
#1. 수정한 파라미터로 모델 만들어서 비교!!!
#2. 수정한 파라미터를 이용해서 파라미터 재조정!!!

############################### 수정한 파라미터 이용 ##################################
#2. 모델
bayesian_params = {
    'max_depth' : [2,10], #default 3/ 0~inf(무한대) / 정수 => 소수점은 정수로 변환하여 적용해야 함
    'gamma': [45,50], #default 0 / 0~inf
    'min_child_weight': [27,30], #default 1 / 0~inf
    'subsample' : [0.9,1], #default 1 / 0~1
    'colsample_bytree' : [0.5,1], #default 1 / 0~1
    'colsample_bylevel' : [0.8,1], #default 1 / 0~1
    'colsample_bynode' : [0.9,1], #default 1 / 0~1
    'reg_alpha' : [26,30], #default 0 / 0~inf / L1 절대값 가중치 규제 / 그냥 alpha도 적용됨
    'reg_lambda' : [2,100], #default 1 / 0~inf / L2 제곱 가중치 규제 / 그냥 lambda도 적용됨
}

def lgb_hamsu(max_depth, gamma, min_child_weight,  
              subsample, colsample_bytree, colsample_bylevel, colsample_bynode, 
              reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 500, 'learning_rate' : 0.02,
        'max_depth' : int(round(max_depth)),                # 무조건 정수
        'gamma' : int(round(gamma)), 
        'min_child_weight' : int(round(min_child_weight)),  
        'subsample' : max(min(subsample, 1), 0),             # 0~1 사이의 값
        'colsample_bytree' : max(min(colsample_bytree, 1), 0),   
        'colsample_bylevel' : max(min(colsample_bylevel, 1), 0),   
        'colsample_bynode' : max(min(colsample_bynode, 1), 0),   
        'reg_lambda' : max(reg_lambda, 0),          # 무조건 양수만
        'reg_alpha' : max(reg_alpha, 0),        
    }
    
    # *여러개의인자를받겠다
    # **키워드를 받겠다(딕셔너리 형태)
    model = XGBRegressor(**params)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    
    return results

lgb_bo = BayesianOptimization(f=lgb_hamsu, 
                              pbounds=bayesian_params,
                              random_state=123,
                              )
lgb_bo.maximize(init_points=5,
                n_iter=50)
print(lgb_bo.max)

# {'target': 0.6192467806110348, 'params': {'colsample_bylevel': 0.9775948096772601, 
# 'colsample_bynode': 0.95225780624897, 'colsample_bytree': 0.552682885472908, 
# 'gamma': 47.72618602567578, 'max_depth': 5.101391780342154, 
# 'min_child_weight': 27.992907357041474, 'reg_alpha': 27.62255887496644, 
# 'reg_lambda': 29.16362781819454, 'subsample': 0.9460444883621733}}