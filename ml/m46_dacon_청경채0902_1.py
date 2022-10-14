from unicodedata import bidirectional
import pandas as pd
import numpy as np
from pytest import yield_fixture
from sklearn.metrics import r2_score
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, GRU, Conv1D, Flatten, Dropout
from keras.layers import LSTMV1, LSTM, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import joblib as jb
import os
import time
from sympy import Min

import tensorflow as tf
tf.random.set_seed(123)


# 1. 데이터
path = 'C:/study/_data/bokchoy/_save/'
train_data, label_data, x_train, y_train, x_val, y_val, x_test, y_test \
= jb.load(path+'m46_save.dat')

# print(train_data.shape, label_data.shape, x_train.shape, y_train.shape,
    #   x_val.shape, y_val.shape, x_test.shape, y_test.shape)
# (195, 1440, 37) (195,) (1607, 1440, 37) (1607,) (206, 1440, 37) (206,) (195, 1440, 37) (195,)

x_train = x_train.reshape(1607, 1440*37)
x_test = x_test.reshape(195, 1440*37)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(1607, 1440, 37)
x_test = x_test.reshape(195, 1440, 37)

# 2. 모델
model = Sequential()
model.add(LSTMV1(32,
               activation='tanh',
               input_shape=(1440,37)))
# model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))
model.summary()

# model =model.load_weights("C:\study\_data/bokchoy\_save\_h5/01_0901_1453_0011-0.2282.h5")

#3. 컴파일, 훈련
model.compile(loss = 'mae', optimizer="adam", metrics=['mse'])

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")   
print(date)

filepath = 'C:/study/_data/bokchoy/_save/_h5/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '01_', date, '_', filename])
                      )

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=12, batch_size=100, 
          validation_data=(x_val, y_val),
          callbacks=[earlyStopping, mcp],
          verbose=1)
end_time = time.time() - start_time

#4. 평가예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)  

#### mse를 rmse로 변환 ####
y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# test_pred -> TEST_ files
for i in range(6):
    thislen=0
    thisfile = 'C:/study/_data/bokchoy/test_target/'+'TEST_0'+str(i+1)+'.csv'
    test = pd.read_csv(thisfile, index_col=False)
    test['rate'] = y_predict[thislen:thislen+len(test['rate'])]
    test.to_csv(thisfile, index=False)
    thislen+=len(test['rate'])


# TEST_파일 취합, 압축파일 생성
import zipfile
import os
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("C:/study/_data/bokchoy/test_target/")
with zipfile.ZipFile("submission_0916_2.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()

# ====================================================== #
# loss :  [0.07051998376846313, 0.008246741257607937]
# RMSE :  0.09081156914046429
# ====================================================== #0000000