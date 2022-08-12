from inspect import Parameter
from tabnanny import verbose
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# 'n_estimators': [100,200,300,400,500,1000]}   #default 100 / 1~inf(무한대) / 정수 
# 'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] #default 0.3/ 0~1 / learning_rate는 eta라고 해도 적용됨
# 'max_depth' : [None, 2,3,4,5,6,7,8,9,10] #default 3/ 0~inf(무한대) / 정수 => 소수점은 정수로 변환하여 적용해야 함
# 'gamma': [0,1,2,3,4,5,7,10,100] #default 0 / 0~inf
# 'min_child_weight': [0,0.01,0.01,0.1,0.5,1,5,10,100] #default 1 / 0~inf
# 'subsample' : [0,0.1,0.2,0.3,0.5,0.7,1] #default 1 / 0~1
# 'colsample_bytree' : [0,0.1,0.2,0.3,0.5,0.7,1] #default 1 / 0~1
# 'colsample_bylevel' : [0,0.1,0.2,0.3,0.5,0.7,1] #default 1 / 0~1
# 'colsample_bynode' : [0,0.1,0.2,0.3,0.5,0.7,1] #default 1 / 0~1
# 'reg_alpha' : [0, 0.1,0.01,0.001,1,2,10] #default 0 / 0~inf / L1 절대값 가중치 규제 / 그냥 alpha도 적용됨
# 'reg_lambda' : [0, 0.1,0.01,0.001,1,2,10] #default 1 / 0~inf / L2 제곱 가중치 규제 / 그냥 lambda도 적용됨


parameters = {'n_estimators': [100, 200],
              'learning_rate' : [0.1, 0.2, 0.3, 0.5],
              'max_depth' : [None, 2,3,4,5,6], #default 6 => 통상 max는 4정도에서 성능이 좋다
              'gamma': [0,1,2,3],
              'min_child_weight': [1,5,10,100],
              'subsample' : [0.7,1],
              'colsample_bytree' : [0.7,1],
              'colsample_bylevel' : [0.7,1],
              'colsample_bynode' : [0.7,1],
              'reg_alpha' : [0, 0.1],
              'reg_lambda' : [0, 0.1],
              }    

#2. 모델
xgb = XGBRegressor(random_state=123,
                    )
model = GridSearchCV(xgb, parameters, verbose=2, cv=kfold, n_jobs=8)

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)    
print('최상의 매개변수 : ', model.best_params_)
print('최상의 점수 : ', model.best_score_)
print('acc : ', result)


#=================================== 결과 =====================================#

#==============================================================================#