import numpy as ny
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
import time


#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=66
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()


#3. 훈련
model.compile(loss='mae', optimizer='adam',
              metrics=['mse']) 

import datetime
date = datetime.datetime.now()      # 2022-07-07 17:21:42.275191
date = date.strftime("%m%d_%H%M")   # 0707_1723
print(date)

filepath = './_ModelCheckPoint/k24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True)  
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, 'k24_', date, '_', filename])
                      )

start_time =time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, 
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1) 
end_time =time.time() - start_time


#4. 평가, 예측
print("=============================== 1. 기본 출력 ================================")
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)


#================================= 1. 기본 출력 ===================================#
# loss :  [1.9813416004180908, 7.096808910369873]
# r2 스코어 :  0.9150926150825165
#=================================================================================#
