
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
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
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target
print(x.shape, y.shape)     # 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1234, train_size=0.8
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
#     model = LGBMClassifier(**params)
    
#     model.fit(x_train, y_train,
#               eval_set=[(x_train, y_train), (x_test, y_test)],
#               eval_metric='acc',
#               verbose=0,
#               early_stopping_rounds=50
#               )
    
#     y_predict = model.predict(x_test)
#     results = accuracy_score(y_test, y_predict)
    
#     return results

# lgb_bo = BayesianOptimization(f=lgb_hamsu, 
#                               pbounds=bayesian_params,
#                               random_state=1234,
#                               )
# lgb_bo.maximize(init_points=5,
#                 n_iter=100)
# print(lgb_bo.max)

# {'target': 0.9473684210526315, 
# 'params': {'colsample_bytree': 1.0, 'max_bin': 19.156222114655804, 
# 'max_depth': 6.0, 'min_child_samples': 85.32632196475512, 
# 'min_child_weight': 1.0, 'num_leaves': 37.701996893102915, 
# 'reg_alpha': 0.01, 'reg_lambda': 0.001, 'subsample': 0.5}}

############################### 수정한 파라미터 이용 ##################################
bayesian_params = {
    'max_depth' : (6, 16), 
    'num_leaves' : (37, 40), 
    'min_child_samples' : (85, 90), 
    'min_child_weight' : (1, 20), 
    'subsample' : (0.5, 1), 
    'colsample_bytree' : (0.7, 1), 
    'max_bin' : (19, 22), 
    'reg_lambda' : (0.001, 0.05), 
    'reg_alpha' : (0.001, 0.05), 
}

def lgb_hamsu(max_depth, num_leaves, min_child_samples, min_child_weight, 
              subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 500, 'learning_rate' : 0.02,
        'max_depth' : int(round(max_depth)),                # 무조건 정수
        'num_leaves' : int(round(num_leaves)), 
        'min_child_samples' : int(round(min_child_samples)), 
        'min_child_weight' : int(round(min_child_weight)),  
        'subsample' : max(min(subsample, 1), 0),                 # 0~1 사이의 값
        'colsample_bytree' : max(min(colsample_bytree, 1), 0),   # 0~1 사이의 값
        'max_bin' : max(int(round(max_bin)), 10),   # 무조건 10이상 
        'reg_lambda' : max(reg_lambda, 0),          # 무조건 양수만
        'reg_alpha' : max(reg_alpha, 0),        
    }
    
    # *여러개의인자를받겠다
    # **키워드를 받겠다(딕셔너리 형태)
    model = LGBMClassifier(**params)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='acc',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    
    return results

lgb_bo = BayesianOptimization(f=lgb_hamsu, 
                              pbounds=bayesian_params,
                              random_state=1234,
                              )
lgb_bo.maximize(init_points=5,
                n_iter=100)
print(lgb_bo.max)

# {'target': 0.9473684210526315, 
# 'params': {'colsample_bytree': 1.0, 'max_bin': 19.341138029640398, 
# 'max_depth': 10.718774518385388, 'min_child_samples': 86.14404924911554, 
# 'min_child_weight': 1.0, 'num_leaves': 37.0, 'reg_alpha': 0.05, 'reg_lambda': 0.05, 
# 'subsample': 0.5}}