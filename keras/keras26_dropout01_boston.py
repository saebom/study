import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(7, activation='linear', input_dim=13))
model.add(Dense(10, activation='linear'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))                                     # 연산 때 20% 비율로 랜덤하게 노드를 없앰
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))                                     # 연산 때 30% 비율로 랜덤하게 노드를 없앰
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))                                     # 연산 때 20% 비율로 랜덤하게 노드를 없앰
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='linear'))
model.add(Dense(1, activation='linear'))
model.summary()

# 함수형 모델
# input1 = Input(shape=(13,))
# dense1 = Dense(7)(input1)
# dense2 = Dense(10, activation='linear')(dense1)
# drop1 = Dropout(0.2)(dense2)
# dense3 = Dense(30, activation='relu')(drop1)
# dense4 = Dense(40, activation='relu')(dense3)
# dense5 = Dense(50, activation='relu')(dense4)
# drop2 = Dropout(0.3)(dense5)
# dense6 = Dense(100, activation='relu')(drop2)
# dense7 = Dense(50, activation='relu')(dense6)
# dense8 = Dense(40, activation='relu')(dense7)
# drop3 = Dropout(0.2)(dense8)
# dense9 = Dense(30, activation='relu')(drop3)
# dense10 = Dense(10, activation='relu')(dense9)
# dense11 = Dense(7, activation='linear')(dense10)
# output1 = Dense(1, activation='linear')(dense11)
# model = Model(inputs=input1, outputs=output1)


#3. 훈련
model.compile(loss='mae', optimizer='adam',
              metrics=['mse']) 

import datetime
date = datetime.datetime.now()      # 2022-07-07 17:21:42.275191
date = date.strftime("%m%d_%H%M")   # 0707_1723
print(date)

filepath = './_ModelCheckPoint/k26/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'


earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True)  
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '01_', date, '_', filename])
                      )

start_time =time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, 
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1) 
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  

#================================= 1. 기본 출력 ===================================#
# loss :  [2.376533031463623, 12.149157524108887]
# r2 스코어:  0.8529462262345302
# k24_0707_1735_0286-1.8825.hdf5
#=================================================================================#

#================================ 2. dorpout 적용 ================================#
# loss :  [2.421360492706299, 12.941900253295898]
# r2 스코어:  0.8433508451819345
# 01_0708_1105_0033-2.7283.hdf5
#=================================================================================#

