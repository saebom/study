
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
#     'max_depth' : (6, 16), 
#     'num_leaves' : (24, 64), 
#     'min_child_samples' : (10, 200), 
#     'min_child_weight' : (1, 20), 
#     'subsample' : (0.5, 1), 
#     'colsample_bytree' : (0.5, 1), 
#     'max_bin' : (10, 500), 
#     'reg_lambda' : (0.001, 10), 
#     'reg_alpha' : (0.01, 50), 
# }

# def lgb_hamsu(max_depth, num_leaves, min_child_samples, min_child_weight, 
#               subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
#     params = {
#         'n_estimators' : 500, 'learning_rate' : 0.02,
#         'max_depth' : int(round(max_depth)),                # 무조건 정수
#         'num_leaves' : int(round(num_leaves)), 
#         'min_child_samples' : int(round(min_child_samples)), 
#         'min_child_weight' : int(round(min_child_weight)),  
#         'subsample' : max(min(subsample, 1), 0),             # 0~1 사이의 값
#         'colsample_bytree' : max(min(colsample_bytree, 1), 0),   
#         'max_bin' : max(int(round(max_bin)), 10),   # 무조건 10이상 
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

# {'target': 0.614937470424126, 
# 'params': {'colsample_bytree': 0.5, 'max_bin': 407.26875362111844, 
# 'max_depth': 16.0, 'min_child_samples': 10.0, 'min_child_weight': 20.0, 
# 'num_leaves': 24.0, 'reg_alpha': 0.01, 'reg_lambda': 10.0, 'subsample': 0.5}}


###################################### [실습] ########################################
#1. 수정한 파라미터로 모델 만들어서 비교!!!
#2. 수정한 파라미터를 이용해서 파라미터 재조정!!!

############################### 수정한 파라미터 이용 ##################################

#2. 모델
bayesian_params = {
    'max_depth' : (16, 26), 
    'num_leaves' : (24, 34), 
    'min_child_samples' : (10, 30), 
    'min_child_weight' : (20, 50), 
    'subsample' : (0.5, 1), 
    'colsample_bytree' : (0.5, 1), 
    'max_bin' : (407, 700), 
    'reg_lambda' : (10, 40), 
    'reg_alpha' : (0.01, 0.1), 
}

def lgb_hamsu(max_depth, num_leaves, min_child_samples, min_child_weight, 
              subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 500, 'learning_rate' : 0.02,
        'max_depth' : int(round(max_depth)),                # 무조건 정수
        'num_leaves' : int(round(num_leaves)), 
        'min_child_samples' : int(round(min_child_samples)), 
        'min_child_weight' : int(round(min_child_weight)),  
        'subsample' : max(min(subsample, 1), 0),             # 0~1 사이의 값
        'colsample_bytree' : max(min(colsample_bytree, 1), 0),   
        'max_bin' : max(int(round(max_bin)), 10),   # 무조건 10이상 
        'reg_lambda' : max(reg_lambda, 0),          # 무조건 양수만
        'reg_alpha' : max(reg_alpha, 0),        
    }
    
    # *여러개의인자를받겠다
    # **키워드를 받겠다(딕셔너리 형태)
    model = LGBMRegressor(**params)
    
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

# {'target': 0.623246923899649, 
# 'params': {'colsample_bytree': 0.5778008132119985, 'max_bin': 462.95635429923664, 
# 'max_depth': 24.34933164101291, 'min_child_samples': 10.573443099944173, 
# 'min_child_weight': 32.17294472758941, 'num_leaves': 28.197217145772086, 
# 'reg_alpha': 0.01758569583352746, 'reg_lambda': 32.230824107188084, 'subsample': 0.6137372171215575}}