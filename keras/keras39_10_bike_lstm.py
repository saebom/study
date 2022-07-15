import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.python.keras.layers import Conv2D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

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
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (9796, 9) (1089, 9) (9796,) (1089,)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(9796, 9, 1) 
x_test = x_test.reshape(1089, 9, 1)
print(x_train.shape)    
print(np.unique(x_train, return_counts=True))


#2. 모델구성
model = Sequential()
model.add(LSTM(100, return_sequences=True, 
               activation='linear', input_shape=(54,1)))
model.add(LSTM(100, return_sequences=False, 
               activation='relu'))   
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))


# #3. 컴파일, 훈련
model.compile(loss = 'mae', optimizer='adam', metrics=['mse'])

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
print(date)

filepath = './_ModelCheckPoint/k24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '10_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)  

#### mse를 rmse로 변환 ####
y_predict = model.predict(x_test)

from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


#=================================== DNN 모델 ====================================#
# loss :  [39.75187683105469, 3872.228759765625]
# R2 :  0.8750965153080439
#=================================================================================#

#=================================== RNN 모델 ====================================#
# loss :  [40.342586517333984, 4253.51708984375]
# R2 :  0.8696968195931143 
#=================================================================================#