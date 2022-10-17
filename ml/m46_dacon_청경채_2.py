import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Input, Dense, GRU, Conv1D, Flatten, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import joblib as jb
import os
import time


# 1. 데이터
path = 'C:/study/_data/bokchoy/_save/'
train_data, label_data, x_train, y_train, x_val, y_val, x_test, y_test \
= jb.load(path+'m46_save.dat')


# 2. 모델
model = Sequential()
model.add(GRU(32, input_shape=(1440,37)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mae', optimizer="adam", metrics=['mse'])

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")   
print(date)

filepath = 'C:/study/_data/bokchoy/_save/_h5/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '01_', date, '_', filename])
                      )

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=100, 
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
with zipfile.ZipFile("submission5.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()
