import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

# import tensorflow as tf
# tf.random.set_seed(777)

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=32
)

# [실습] 만들어서 실행

print(x.shape, y.shape) # (13, 3) (13,)
x = x.reshape(13, 3, 1)
print(x.shape) # (13, 3, 1)


#2. 모델구성
model = Sequential()
model.add(Bidirectional(LSTM(units=128, return_sequences=True, activation='relu'), input_shape=(3, 1)))      
model.add(Bidirectional(LSTM(128)))
model.add(Dense(128, activation='relu'))         
model.add(Dense(64, activation='relu'))         
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

earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '01_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x, y, epochs=100, batch_size=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time



#4. 평가, 예측 
loss = model.evaluate(x, y)
y_predict = np.array([50,60,70]).reshape(1,3,1)
result = model.predict(y_predict)
print('loss : ', loss)
print('[50, 60, 70]의 결과 : ', result)

# loss :  8.88656234741211
# [50, 60, 70]의 결과 :  [[85.94356]]
