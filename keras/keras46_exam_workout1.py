import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import tensorflow as tf
print(tf.__version__)   # 2.8.2


#1. 데이터
path = './_data/test_amore_0718/'
amore = pd.read_csv(path +"아모레220718.csv", thousands=",", encoding='cp949')  # 마이크로소프트 'cp949' / 맥과 리눅스는 'utf8'

print(amore.shape)    # (3180, 17)

print(amore.describe)
print(amore.head(5))
print(amore.isnull().sum())
print(amore.info())

#일자 변환
amore['일자'] = pd.to_datetime(amore['일자'], infer_datetime_format=True)

#과거에서 현재 순으로 행을 역순 시켜줌
amore = amore.loc[::-1].reset_index(drop=True)
amore = amore.drop(range(0, 2062), axis=0)
amore = amore.drop(columns=['전일비'], axis=1)

print(amore.head(5))
print(amore.describe)
print(amore.shape)  # (1118, 16)

amore = amore.apply(pd.to_numeric) # convert all columns of DataFrame
print(amore.info())
print(amore.head(5))
print(amore.isnull().sum())

x = amore.drop(columns=['일자', '시가', '금액(백만)', '신용비', '외인(수량)', '프로그램', '외인비'], axis=1) 
x = np.array(x)
print(x.shape) # (1118, 9)

y = amore['시가']
print(y.shape) # (1118,)

time_steps = 5
y_column = 5

def split_xy(dataset, time_steps, y_column):                 
    x = []
    y = []
    for i in range(len(dataset)):
        x_end_number = i + time_steps      # 0 + 5   > 5
        y_end_number = x_end_number + y_column - 1    # 5 + 3 -1 > 7
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, 0]                   # 0 : 5 , : -1   > 0행~4행, 마지막열 뺀 전부
        tmp_y = dataset[x_end_number-1:y_end_number, 0]       # 5 - 1 : 7 , -1  > 마지막 열의 4~6행
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy(x, time_steps, y_column)
y_predict = y[-1]
print(x.shape, y.shape) # (1110, 5) (1110, 5)
print(y_predict)        # [131000. 137500. 138000. 136000. 136500.]

# 스케일링
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
scaler.fit(x)
x1 = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=66)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (666, 5) (444, 5) (666, 5) (444, 5)

x_train = x_train.reshape(666, 5, 1) 
x_test = x_test.reshape(444, 5, 1)
print(x_train.shape)    
print(np.unique(x_train, return_counts=True))


#2. 모델구성
model = Sequential()
model.add(LSTM(100, return_sequences=True, 
               activation='linear', input_shape=(5,1)))
model.add(LSTM(100, return_sequences=False, 
               activation='relu'))   
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")   
print(date)

filepath = './_ModelCheckPoint/k46/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '_', date, '_', filename])
                      )
start_time = time.time() 
history = model.fit(x_train, y_train, epochs=500, batch_size=128,
                    validation_split=0.2, callbacks=[earlyStopping],
                    verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss :', loss) 

y_pred = np.array(y_predict).reshape(1, 5, 1)
result = model.predict(y_pred)
print('[7월 19일 시가] 예측값 : ', result)

import matplotlib
import matplotlib.pyplot as plt2

plt2.plot(y_predict, color='red', label='Prediction')
plt2.plot(y_test, color='blue', label='Ground Truth')
plt2.legend(loc='upper left')
plt2.show()

# loss : 123147392.0
# [7월 19일 시가] 예측값 :  [[133022.23]]


# model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 5, 100)            40800
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 100)               80400
# _________________________________________________________________
# dense (Dense)                (None, 100)               10100
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               10100
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 101
# =================================================================
# Total params: 141,501
# Trainable params: 141,501
# Non-trainable params: 0
