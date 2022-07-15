import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import Conv1D, Flatten, MaxPooling1D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv',   # 예측에서 사용!!
                       index_col=0)

#### 결측치 처리 ####
test_set = test_set.fillna(method='ffill')
train_set = train_set.dropna()  # nan 값 삭제

x = train_set.drop(['count'], axis=1)
y = train_set['count']
print(x.shape, y.shape)  # (1328, 9) (1328,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=66
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (1062, 9) (266, 9) (1062,) (266,)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(1062, 3*3, 1)
x_test = x_test.reshape(266, 3*3, 1)
print(x_train.shape)    
print(np.unique(x_train, return_counts=True))

#2. 모델구성 
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, padding='same', 
                 activation='relu', input_shape=(3*3, 1)))
model.add(Dropout(0.25))     
model.add(Conv1D(64, 3, padding='same', activation='relu'))                
model.add(Dropout(0.25))     
model.add(Conv1D(128, 3, padding='same', activation='relu'))
model.add(Dropout(0.4))     
model.add(Conv1D(128, 3, padding='same', activation='relu'))   
model.add(Dropout(0.25))                 
model.add(Conv1D(64, 3, padding='same', activation='relu'))                
model.add(Dropout(0.2))   
model.add(Conv1D(32, 3, padding='same', activation='relu'))                
model.add(Dropout(0.2))   

model.add(Flatten())   
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.summary()



#3. 컴파일, 훈련
model.compile(loss = 'mae', optimizer="adam", metrics=['mse'])

import datetime
date = datetime.datetime.now()      # 2022-07-07 17:21:42.275191
date = date.strftime("%m%d_%H%M")   # 0707_1723
print(date)

filepath = './_ModelCheckPoint/k41/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '09_', date, '_', filename])
                      )

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=5000, batch_size=100, 
          validation_split=0.2,
          callbacks=[earlyStopping],
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


# print("=====================================================================")
print("걸린시간 : ", end_time)


#===================================== DNN  ======================================#
# loss :  [40.44511795043945, 3217.836669921875]
# R2 :  0.4822973526333515 
#=================================================================================#

#==================================== ConvD1 =====================================#
# loss :  
# R2 :  
#=================================================================================#