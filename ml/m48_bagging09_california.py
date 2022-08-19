import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel



#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

x = np.delete(x, [3, 4], axis=1)
print(x.shape)  # (20640, 6)

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
model = BaggingRegressor(DecisionTreeRegressor(), 
# model = BaggingRegressor(XGBRegressor(), 
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
print('Bagging_XGBRegressor 결과 : ', result)
print('걸린 시간 : ', end - start)


#=================================== 결과 =====================================#
# 기존  r2 :  0.8437110641109735  
# Bagging_XGBRegressor 결과 :  0.8569589811696056
# 걸린 시간 :  93.93555235862732 
#==============================================================================#