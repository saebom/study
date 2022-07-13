import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN
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

# x의 shape = (행, 열, 몇 개씩 자르는지(timesteps)!!!) => timesteps
x = x.reshape(7, 3, 1)
print(x.shape) # (7, 3, 1)


#2. 모델구성
model = Sequential()
model.add(SimpleRNN(15, input_shape=(3, 1)))    #  [batch, timesteps, feature] 
# model.add(SimpleRNN(32))    # ValueError: Input 0 of layer simple_rnn_1 is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (None, 64)
model.add(Dense(5, activation='relu'))         # 2차원의 output이 나오기 때문에 Dense로 받아줌
model.add(Dense(1))
model.summary()

# units*(feature)
'''
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
'''


# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# simple_rnn (SimpleRNN)       (None, 10)                120
# _________________________________________________________________
# dense (Dense)                (None, 5)                 55
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 6
# =================================================================
# Total params: 181
# Trainable params: 181
# Non-trainable params: 0

# ==> Param 결과해석
# recurrent_weights + input_weights + biases
# (units*units) + (units*features) + units
# units*(units+features) + units
# 10(10+1) + 10 = 120







# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# simple_rnn (SimpleRNN)       (None, 15)                255
# _________________________________________________________________
# dense (Dense)                (None, 5)                 80
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 6
# =================================================================
# Total params: 341
# Trainable params: 341
# Non-trainable params: 0

# ==> Param 결과해석
# recurrent_weights + input_weights + biases
# (units*units) + (units*features) + units
# units*(units+features) + units
# 15(15+1) + 15 = 120



