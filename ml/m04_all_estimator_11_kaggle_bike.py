import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler


#1. 데이터
path = './_data/bike/'
train_set = pd.read_csv(path + 'train.csv')
print(train_set)
print(train_set.shape)   # (10886, 11)
print(train_set.columns) # ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                         #  'humidity', 'windspeed', 'casual', 'registered', 'count']
                        
print(train_set.info())
print(train_set.describe())
print(train_set.isnull().sum())

test_set = pd.read_csv(path + 'test.csv')
print(test_set)
print(test_set.shape)   # (6493, 8)
print(test_set.columns) # ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                        #  'humidity', 'windspeed']   ==> casual, registered, count
print(test_set.info())
print(test_set.describe())
print(test_set.isnull().sum())  

#### 결측치 처리 ####
train_set = train_set[train_set['weather'] != 4]  # weather가 4인 데이터는 이상치(폭우, 폭설이 내리는 날 저녁 6시에 대여)

from datetime import datetime
train_set['datetime'] = pd.to_datetime(train_set['datetime'])
test_set['datetime'] = pd.to_datetime(test_set['datetime'])

# 날짜 feature 생성

L = ['year', 'month', 'date', 'hour', 'weekday']
train_set = train_set.join(pd.concat([getattr(train_set['datetime'].dt, i).rename(i) for i in L], axis=1))
test_set = test_set.join(pd.concat([getattr(test_set['datetime'].dt, i).rename(i) for i in L], axis=1))

print(train_set)
print(test_set)

# drop_features

drop_features = ['casual', 'registered', 'datetime', 'year', 'date', 'windspeed', 'month']
train_set = train_set.drop(drop_features, axis=1)
test_set = test_set.drop(['datetime', 'year', 'date', 'windspeed', 'month'], axis=1)

print("train_set.columns :", train_set.columns)
x = train_set.drop(['count'], axis=1)
print(x)
print(x.columns)
# print(x.shape)  

y = train_set['count']
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=13
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성

model = LinearSVR(


#3. 컴파일, 훈련

model.fit(x_train, y_train)

#4. 평가예측

results = model.score(x_test, y_test)
print('결과 r2 : ', results)




#===================================== 결  과 ==========================================#
# LinearSVR() 결과 r2 :  0.27569630565541614
# LinearRegression() 결과 r2 :  
# KNeighborsRegressor() 결과 r2 :  
# DecisionTreeRegressor() 결과 결과 r2 :  
# RandomForestRegressor() 결과 r2 : 
#=======================================================================================#

