from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

import tensorflow as tf
tf.random.set_seed(777)

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

# loss의 스케일 조정을 위해 0 ~ 255 -> 0 ~ 1 범위로 만들어줌
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

mean = np.mean(x_train, axis=(0 , 1 , 2 , 3))
std = np.std(x_train, axis=(0 , 1 , 2 , 3))
x_train = (x_train-mean)/std
x_test = (x_test-mean)/std

x_train = x_train.reshape(50000, 32*32, 3)
x_test = x_test.reshape(10000, 32*32, 3)
print(x_train.shape)    # (50000, 32, 32, 3)
print(np.unique(x_train, return_counts=True))

# One Hot Encoding
from tensorflow.python.keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)


#2. 모델링 
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(32*32, 3)))
model.add(MaxPooling1D())  
model.add(Dropout(0.2))     
model.add(Conv1D(32, 3, activation='relu'))                
model.add(MaxPooling1D())  
model.add(Dropout(0.2))     
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D())     
model.add(Dropout(0.2))     
model.add(Conv1D(64, 3, activation='relu'))   
model.add(MaxPooling1D())     
model.add(Dropout(0.2))                    
model.add(Conv1D(128, 3, padding='same', activation='relu'))  
             
model.add(Flatten())    
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='Nadam', 
              metrics=['accuracy'])

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
print(date)

filepath = './_ModelCheckPoint/k41/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '02_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=128,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
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



#==================================== Conv2D ======================================#
# loss :  0.6298010945320129
# accuracy :  0.786300003528595  
# 걸린시간 :  409.54772663116455
#===================================================================================#

#==================================== Conv1D ======================================#
# loss :  0.980301558971405
# accuracy :  0.6628999710083008
# 걸린시간 :  130.10892176628113
#===================================================================================#