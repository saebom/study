import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets['target']

print(x.shape, y.shape) # (506, 13) (506,)
x = x.reshape(506, 13, 1)
print(x.shape) # (506, 13, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

#2. 모델 구성
model = Sequential()
model.add(LSTM(units=7, return_sequences=True, 
               activation='relu', input_shape=(13,1)))
model.add(LSTM(10, return_sequences=False, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='linear'))
model.add(Dense(1, activation='linear'))
model.summary() 


#3. 훈련
model.compile(loss='mae', optimizer='adam',
              metrics=['mse']) 

import datetime
date = datetime.datetime.now()      # 2022-07-07 17:21:42.275191
date = date.strftime("%m%d_%H%M")   # 0707_1723
print(date)

filepath = './_ModelCheckPoint/k39/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'


earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True)  
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '01_', date, '_', filename])
                      )

start_time =time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=1, 
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1) 
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)  
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  

#================================= DNN 모델 ======================================#
# loss :  [2.376533031463623, 12.149157524108887]
# r2 스코어:  0.8529462262345302
#=================================================================================#

#================================== RNN 모델 =====================================#
# loss :  [3.4893016815185547, 24.739513397216797]
# r2 스코어:  0.7005521401941861
#=================================================================================#

