from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28*28, 1)  # (60000, 28, 28) (60000,)
x_test = x_test.reshape(10000, 28*28, 1)   # (10000, 28, 28) (10000,)
# print(x_train.shape)    # (60000, 28, 28)
# print(np.unique(x_train, return_counts=True))


# One Hot Encoding
import pandas as pd
# df = pd.DataFrame(y)
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train)
print(y_train.shape)


#2. 모델링 
model = Sequential()
model.add(Conv1D(32, 4, input_shape=(28*28, 1)))      
model.add(MaxPooling1D())           
model.add(Dropout(0.2))
model.add(Conv1D(64, 4, activation='relu'))                
model.add(MaxPooling1D())          
model.add(Dropout(0.2))
model.add(Conv1D(64, 4, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())    # (N, 63)  (N, 175)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
print(date)

filepath = './_ModelCheckPoint/k41/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '01_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=50, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)
print('걸린시간 : ', end_time)

import tensorflow as tf
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)    
y_test = tf.argmax(y_test, axis=1)          



#==================================== Conv2D ======================================#
# loss :  0.033741943538188934
# accuracy :  0.9915000200271606
# 걸린시간 :  313.09232115745544
#===================================================================================#

#==================================== Conv1D ======================================#
# loss :  0.06594347208738327
# accuracy :  0.9835000038146973
# 걸린시간 :  395.50878143310547
#===================================================================================#