from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, MaxPooling2D
from tensorflow.python.keras.layers import Conv1D, Conv2D, LSTM, Reshape
from keras.datasets import mnist
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)  # (60000, 28, 28) (60000,)
x_test = x_test.reshape(10000, 28, 28, 1)   # (10000, 28, 28) (10000,)
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
model.add(Conv2D(filters=64, kernel_size=(3, 3),    
                 padding='same', input_shape=(28, 28, 1)))      
model.add(MaxPooling2D())                                           # (N, 14, 14, 64)      
model.add(Conv2D(32, (3, 3)))                                       # (N, 12, 12, 32)                
model.add(Conv2D(7, (3, 3)))                                        # (N, 10, 10, 7)
model.add(Reshape(target_shape=(700,)))                             # (N, 700)
# model.add(Flatten())
model.add(Dense(100, activation='relu'))                            # (N, 100)  
# model.add(Dense(10, activation='relu'))                                            
model.add(Reshape(target_shape=(100, 1)))                           # (N, 100, 1)
model.add(Conv1D(filters=10, kernel_size=3))                        # (N, 98, 10)
model.add(LSTM(16))                                                 # (N, 16)
model.add(Dense(32, activation='relu'))                             # (N, 32)
model.add(Dense(10, activation='softmax'))                          # (N, 10)
model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
print(date)

filepath = './_ModelCheckPoint/k28/'
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
                 callbacks=[earlyStopping, mcp],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

import tensorflow as tf
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)    
y_test = tf.argmax(y_test, axis=1)          



#=============================== loss, accuracy ====================================#
# loss :  0.04272077605128288
# accuracy :  0.9915000200271606
#===================================================================================#

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 64)        640
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 12, 12, 32)        18464
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 10, 10, 7)         2023
# _________________________________________________________________
# flatten (Flatten)            (None, 700)               0
# _________________________________________________________________
# dense (Dense)                (None, 100)               70100
# _________________________________________________________________
# reshape (Reshape)            (None, 100, 1)            0
# _________________________________________________________________
# conv1d (Conv1D)              (None, 98, 10)            40
# _________________________________________________________________
# lstm (LSTM)                  (None, 16)                1728
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                544
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                330
# =================================================================
# Total params: 93,869
# Trainable params: 93,869
# Non-trainable params: 0