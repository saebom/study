import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Bidirectional
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
x = x.reshape(7, 3, 1)
print(x.shape) # (7, 3, 1)


#2. 모델구성
model = Sequential()
model.add(SimpleRNN(15, input_shape=(3, 1), return_sequences=True))
model.add(Bidirectional(SimpleRNN(10)))
model.add(Dense(128, activation='relu'))         
model.add(Dense(128, activation='relu'))         
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


# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 3, 10)             120

#  bidirectional (Bidirectiona  (None, 10)               160
#  l)
# =================================================================
# bidirectional Param 계산 
# ==> 2 X [(HiddenUnit+inputSize)XHuddenUnits+HuddenUnits]
# inputSize 10, unit 5일때  2X[5X(5+10)+5] = 160


# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 3, 10)             120

#  bidirectional (Bidirectiona  (None, 20)               420
# =================================================================
# inputSize 10, unit 10일 때 2X[10X(10+10)+10]


# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 3, 10)             120

#  bidirectional (Bidirectiona  (None, 30)               780
#  l)
# =================================================================
# inputSize 10, unit 15일 때 2X[15X(15+10)+15] = 780

 
#  _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 3, 10)             120

#  bidirectional (Bidirectiona  (None, 50)               1800
#  l)
# =================================================================
# inputSize 10, unit 25일 때 2*[25*(25+10)+25] = 1800


# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 3, 15)             255

#  bidirectional (Bidirectiona  (None, 20)               520
#  l)
# =================================================================
# inputSize 15, unit 10일 때 
# 첫번째 param => 15*(15+1)+15 = 255
# 두번째 param => 2*[10*(10+15)+10] = 520