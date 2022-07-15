import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time


import tensorflow as tf
tf.random.set_seed(72)

#1. 데이터 
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])    # 시계열 데이터이므로 현재수업차시에서는 임의로 자름
# y = ?

x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]])
y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape) # (7, 3) (7,)

# x의 shape = (행, 열, 몇개씩 자르는지!!!)
x = x.reshape(7, 3)
print(x.shape) # (7, 3)


#2. 모델구성
model = Sequential()
# model.add(LSTM(10, input_shape=(3, 1), return_sequences=False))
model.add(Conv1D(10, 2, input_shape=(3, 1)))
model.add(Flatten())    
model.add(Dense(32, activation='relu'))         
model.add(Dense(1))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)


#4. 평가, 예측 
loss = model.evaluate(x, y)
y_pred = np.array([8, 9, 10]).reshape(1, 3, 1)  # [[[8], [9], [10]]]
result = model.predict(y_pred)
print('loss : ', loss)
print('[8, 9, 10]의 결과 : ', result)

# loss :  0.015911774709820747
# [8, 9, 10]의 결과 :  [[11.042677]]
