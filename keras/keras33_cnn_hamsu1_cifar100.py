from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.datasets import cifar100
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

import tensorflow as tf
tf.random.set_seed(777)

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

# scale (이미지를 0 ~ 255 -> 0 ~ 1 범위로 만들어줌)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32' )
x_train = x_train/255
x_test = x_test/255

mean = np.mean(x_train, axis=(0 , 1 , 2 , 3))
std = np.std(x_train, axis=(0 , 1 , 2 , 3))
x_train = (x_train-mean)/std
x_test = (x_test-mean)/std

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)
print(x_train.shape)    # (50000, 32, 32, 3)
print(np.unique(x_train, return_counts=True))

# One Hot Encoding
from tensorflow.python.keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)


#2. 모델링 
model = Sequential()
input1 = Input(shape=(32, 32, 3))
dense1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', 
                 activation='relu')(input1)
dense2 = MaxPooling2D(2, 2)(dense1)  
dense3 = Dropout(0.25)(dense2)     
dense4 = Conv2D(64, (3, 3), padding='same', activation='relu')(dense3)                
dense5 = MaxPooling2D(2, 2)(dense4)  
dense6 = Dropout(0.25)(dense5)     
dense7 = Conv2D(128, (3, 3), padding='same', activation='relu')(dense6)
dense8 = MaxPooling2D(2, 2)(dense7)     
dense9 = Dropout(0.4)(dense8)
dense10 = Conv2D(254, (3, 3), padding='same', activation='relu')(dense9)
dense11 = MaxPooling2D(2, 2)(dense10)
dense12 = Dropout(0.4)(dense11)
             
dense14 = Flatten()(dense12)   
dense15 = Dense(128, activation='relu')(dense14)
dense16 = Dropout(0.2)(dense15)
output1 = Dense(100, activation='softmax')(dense16)
model = Model(inputs=input1, outputs=output1)
model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='Nadam', 
              metrics=['accuracy'])

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
print(date)

filepath = './_ModelCheckPoint/k28/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '03_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=300, batch_size=128,
                 validation_split=0.3,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

y_predict = model.predict(x_test)
y_predict = y_predict.argmax(axis=1)      
y_test = y_test.argmax(axis=1)   

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
# loss :  1.872557282447815
# accuracy :  0.5112000107765198
#================================================================================#


#=============================== 함수형 결과값 ===================================#
# loss :  1.8275066614151
# accuracy :  0.5164999961853027 
#================================================================================#

