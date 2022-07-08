from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
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
print(x_train.shape)    # (60000, 28, 28)
print(np.unique(x_train, return_counts=True))
print(np.unique(y_train, return_counts=True))



# [실습] 
# acc 0.98 이상
# convolution 3개 이상 사용
import pandas as pd
# df = pd.DataFrame(y)
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train)
print(y_train.shape)



#2. 모델링 
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4, 4),    # 출력 : (N, 28, 28, 64) => padding = 'same'이므로
                 padding='same', 
                 input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))           # 출력 : (N, 14, 14, 64)
model.add(Conv2D(64, (4, 4), 
                 padding='valid',                   # padding = 'valid' 가 default
                 activation='relu'))                # 출력 : (N, 3, 3, 7)   (N, 5, 5, 7)   (N, 4, 4, 7)
model.add(MaxPooling2D(2, 2))           # 출력 : (N, 14, 14, 64)
model.add(Conv2D(64, (4, 4), 
                 padding='same',                   # padding = 'valid' 가 default
                 activation='relu'))
model.add(Flatten())    # (N, 63)  (N, 175)
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()


# #3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy', 'mse'])

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
print(date)

filepath = './_ModelCheckPoint/k28/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '01_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
acc = model.evaluate(x_test, y_test)
# acc = model.evaluate(x_test, y_test)
print('accuracy : ', acc)

import tensorflow as tf
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)    # pandas에서 사용 : get_dummies
y_test = tf.argmax(y_test, axis=1)          # pandas에서 사용 : get_dummies




#=========================================== 내용 정리 ======================================================#
#  loss :  0.07398068159818649
#  accuracy 0.9861999750137329
# accuracy :  [0.06225131079554558, 0.9866999983787537, 0.002123833168298006]
#==============================================================================================================#
