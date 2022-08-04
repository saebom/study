import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression, LinearRegression     #LogisticRegression은 분류모델, LinearRegression 회귀모델
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model = LinearSVR()
# model = LinearRegression()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가예측
results = model.score(x_test, y_test)
print('결과 r2 : ', results)



#===================================== 결  과 ==========================================#
# LinearSVR() 결과 r2 :  
# LinearRegression() 결과 r2 :  
# KNeighborsRegressor() 결과 r2 :  
# DecisionTreeRegressor() 결과 결과 r2 :  
# RandomForestRegressor() 결과 r2 :  
#=======================================================================================#

