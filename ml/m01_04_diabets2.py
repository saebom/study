import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
# from tensorflow.python.keras.models import Sequential, Model, load_model
# from tensorflow.python.keras.layers import Dense, Input, Dropout
# from tensorflow.python.keras.layers import Conv1D, Flatten
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.svm import LinearSVR

import tensorflow as tf
tf.random.set_seed(777)

#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=72
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (309, 10) (133, 10) (309,) (133,)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# x_train = x_train.reshape(309, 2*5, 1) 
# x_test = x_test.reshape(133, 2*5, 1)
# print(x_train.shape)    
# print(np.unique(x_train, return_counts=True))


#2. 모델 구성
# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=1, padding='same', 
#                  activation='relu', input_shape=(2*5, 1)))
# model.add(Dropout(0.25))     
# model.add(Conv1D(64, 1, padding='same', activation='relu'))                
# model.add(Dropout(0.25))     
# model.add(Conv1D(128, 1, padding='same', activation='relu'))
# model.add(Dropout(0.4))     
# model.add(Conv1D(128, 1, padding='same', activation='relu'))   
# model.add(Dropout(0.25))                 
# model.add(Conv1D(64, 1, padding='same', activation='relu'))                
# model.add(Dropout(0.2))   
# model.add(Conv1D(32, 1, padding='same', activation='relu'))                
# model.add(Dropout(0.2))   

# model.add(Flatten())   
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='linear'))
# model.summary()
model = LinearSVR()


#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])  

# import datetime
# date = datetime.datetime.now()      
# date = date.strftime("%m%d_%H%M")   
# print(date)

# filepath = './_ModelCheckPoint/k41/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', 
#                               verbose=1, 
#                               restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
#                       save_best_only=True, 
#                       filepath="".join([filepath, '_03_', date, '_', filename])
#                       )

# start_time = time.time() 
# hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
#                  validation_split=0.2,
#                  callbacks=[earlyStopping],
#                  verbose=1)
# end_time = time.time() - start_time
model.fit(x_train, y_train)


#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)

# result = model.score(x_test, y_test)
# print('결과 r2 : ', result)
results = model.score(x_test, y_test)
print('결과 r2 : ', results)

# y_predict = model.predict(x_test)  
# r2 = r2_score(y_test, y_predict)
# print('r2 스코어: ', r2)  

# print("=====================================================================")
# print("걸린시간 : ", end_time)




#================================ SVM 적용 결과 ===================================#
# 결과 r2 :  0.15523905134370397
# =================================================================================
# loss :  [2620.469970703125, 2620.469970703125]
# r2 스코어:  0.5625893628075687
# 걸린시간 :  10.755416870117188
#==================================================================================#