import numpy as np
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=72, train_size=0.8
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor

# model = BaggingRegressor(LinearRegression(), 
# model = BaggingRegressor(DecisionTreeRegressor(), 
model = BaggingRegressor(XGBRegressor(), 
                          n_estimators=100,
                          n_jobs=-1,
                          random_state=72
                          )

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()


#4. 평가, 예측
result = model.score(x_test, y_test)    
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('Bagging_XGBRegressor 결과 : ', result)
print('걸린 시간 : ', end - start)

#=================================== 결과 =====================================#
# 기존 : model.score :  0.5736232931020658
# Bagging_XGBRegressor 결과 :  0.5541852675484074
# 걸린 시간 :  3.84659743309021
#==============================================================================#