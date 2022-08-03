import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
# from tensorflow.python.keras.models import Sequential, Model, load_model
# from tensorflow.python.keras.layers import Dense, LSTM, Dropout
# from tensorflow.python.keras.layers import Conv2D, Flatten
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.svm import LinearSVC

import tensorflow as tf
tf.random.set_seed(777)

#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target

print(x.shape, y.shape) # (442, 10) (442,)
x = x.reshape(442, 10, 1)
print(x.shape) # (442, 10, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=72
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (309, 10, 1) (133, 10, 1) (309,) (133,)


#2. 모델 구성
# model = Sequential()
# model.add(LSTM(100, return_sequences=True, 
#                activation='relu', input_shape=(10,1)))
# model.add(LSTM(100, return_sequences=False, activation='relu'))
# model.add(Dropout(0.3))                                     # 연산 때 30% 비율로 랜덤하게 노드를 없앰
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))                                     # 연산 때 20% 비율로 랜덤하게 노드를 없앰
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1, activation='linear'))
model = LinearSVC()


#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])  

# import datetime
# date = datetime.datetime.now()      
# date = date.strftime("%m%d_%H%M")   
# print(date)

# filepath = './_ModelCheckPoint/k39/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', 
#                               verbose=1, 
#                               restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
#                       save_best_only=True, 
#                       filepath="".join([filepath, '_03_', date, '_', filename])
#                       )

# start_time = time.time() 
# hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
#                  validation_split=0.2,
#                  callbacks=[earlyStopping],
#                  verbose=1)
# end_time = time.time() - start_time
model.fit(x_train, y_train)

#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)

# y_predict = model.predict(x_test)  
# r2 = r2_score(y_test, y_predict)
# print('r2 스코어: ', r2)  
result = model.score(x_train, y_train)
print('결과 r2 : ', result)

#================================= DNN 모델 ======================================#
# loss :  [2643.515869140625, 2643.515869140625]
# r2 스코어:  0.5587424215329446
#=================================================================================#

#================================== RNN 모델 =====================================#
# loss :  [4174.38720703125, 4174.38720703125]
# r2 스코어:  0.3032083776555
#=================================================================================#

