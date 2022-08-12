import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor



#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(datasets.DESCR)
print(x.shape)  # (581012, 54)

# 스케일링
from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
x = scaler.fit_transform(x)

le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72, stratify=y
    )


n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

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
# parameters = {'n_estimators': [100],
#               'learning_rate' : [0.1],
#               'max_depth' : [3], #default 6 => 통상 max는 4정도에서 성능이 좋다
#               'gamma': [1],
#               'min_child_weight': [1],
#               'subsample' : [1],
#               'colsample_bytree' : [1],
#               'colsample_bylevel' : [1],
#               'colsample_bynode' : [1],
#               'reg_alpha' : [0],
#               'reg_lambda' : [1],
#               }  


#2. 모델
# xgb = XGBClassifier(random_state=123,
#                     )
xgb = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)

model = GridSearchCV(xgb, parameters, verbose=2, cv=kfold, n_jobs=8)


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)    
print('최상의 매개변수 : ', model.best_params_)
print('최상의 점수 : ', model.best_score_)
print('acc : ', result)


#=================================== 결과 =====================================#
# 최상의 매개변수 :  {'colsample_bylevel': 1, 'colsample_bynode': 1, 
# 'colsample_bytree': 1, 'gamma': 2, 'learning_rate': 0.2, 'max_depth': 5, 
# 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0, 'reg_lambda': 0, 'subsample': 0.7}
# 최상의 점수 :  0.6536859089436121
# acc :  0.6576052337683306
#==============================================================================#