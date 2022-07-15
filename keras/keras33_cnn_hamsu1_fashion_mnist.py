import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.datasets import fashion_mnist
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
import time

import tensorflow as tf
tf.random.set_seed(66)

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)  # (60000, 28, 28) (60000,)
x_test = x_test.reshape(10000, 28, 28, 1)    # (10000, 28, 28) (10000,)
# print(x_train.shape)    # (60000, 28, 28)
# print(np.unique(x_train, return_counts=True))

import pandas as pd
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train)
print(y_train.shape)


#2. 모델링 
model = Sequential()

input1 = Input(shape=(28, 28, 1))
dense1 = Conv2D(64, kernel_size=(4, 4), padding='same')(input1)
dense2 = MaxPooling2D(2, 2)(dense1)
dense3 = Dropout(0.2)(dense2)
dense4 = Conv2D(64, (4, 4), padding='valid', activation='relu')(dense3)
dense5 = MaxPooling2D(2, 2)(dense4)         
dense6 = Dropout(0.2)(dense5)
dense7 = Conv2D(64, (4, 4), padding='same', activation='relu')(dense6)
dense8 = MaxPooling2D(2, 2)(dense7)         
dense9 = Dropout(0.2)(dense8)
dense10 = Flatten()(dense9)    # (N, 63)  (N, 175)
dense11 = Dense(128, activation='relu')(dense10)
dense12 = Dropout(0.2)(dense11)
dense13 = Dense(64, activation='relu')(dense12)
output1= Dense(10, activation='softmax')(dense13)
model = Model(inputs=input1, outputs=output1)
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

earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '04_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=128,
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

acc = accuracy_score(y_test, y_predict)
print("=====================================================================")   
print('acc 스코어 : ', acc)  

print("=====================================================================")
print("걸린시간 : ", end_time)


#그래프로 비교
font_path = 'C:\Windows\Fonts\malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()    
plt.title('로스값과 검증로스값')    
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()   
plt.show()


#=============================== 이전 결과값 ====================================#
# loss :  0.28520235419273376
# accuracy :  0.9143000245094299
#================================================================================#


#=============================== 함수형 결과값 ===================================#
# loss :  
# accuracy :  
#================================================================================#
