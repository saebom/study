import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time


import tensorflow as tf
tf.random.set_seed(66)

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
model.add(GRU(units=100, input_shape=(3, 1)))    #  [batch, timesteps, feature]   
model.add(Dense(100, activation='relu'))         
model.add(Dense(100, activation='relu'))         
model.add(Dense(100, activation='relu'))         
# model.add(Dense(32, activation='relu'))         
model.add(Dense(1))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
print(date)

filepath = './_ModelCheckPoint/k35/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '01_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x, y, epochs=300, batch_size=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time



#4. 평가, 예측 
loss = model.evaluate(x, y)
y_pred = np.array([8, 9, 10]).reshape(1, 3, 1)  # [[[8], [9], [10]]]
result = model.predict(y_pred)
print('loss : ', loss)
print('[8, 9, 10]의 결과 : ', result)

# loss :  0.18099834024906158
# [8, 9, 10]의 결과 :  [[10.23832]]


# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# gru (GRU)                    (None, 100)               30600
# _________________________________________________________________
# dense (Dense)                (None, 100)               10100
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               10100
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 101
# =================================================================
# Total params: 50,901
# Trainable params: 50,901
# Non-trainable params: 0

# ================================ 내용정리 =====================================#
# [simpleRNN] uints: 10 ==> 10*(10+1)+10 = 120

# [LSTM] units : 10 ==> 4*(10*(10+1)+10) = 480
#                       4*(20*(20+1)+10) = 1760
# 결론 : LSTM = simpleRNN * 4
# 숫자 4의 의미는 cell state, input gate, output gate, forget gate

# [GRU] units : 100 ==> 3*(100(100+1)+100) = 30600
# 결론 : LSTM = simpleRNN * 4
# 숫자 3의 의미는 hidden state, reset gate, update gate
