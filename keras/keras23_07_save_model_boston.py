import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(7, activation='linear', input_dim=13))
model.add(Dense(10, activation='linear'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='linear'))
model.add(Dense(1, activation='linear'))

# model.save("./_save/keras23_7_save_model1.h5")
# model.save_weights("./_save/keras23_7_save_weights1.h5")

# model = load_model("./_save/keras23_7_save_model1.h5")
# model.load_weights('./_save/keras23_7_save_weights1.h5')
# model.load_weights('./_save/keras23_7_save_weights2.h5')


#3. 훈련
# model.compile(loss='mae', optimizer='adam',
#               metrics=['mse']) 
# earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
#                               restore_best_weights=True)  

# start_time =time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, 
#                  validation_split=0.2,
#                  callbacks=[earlyStopping],
#                  verbose=1) 
# end_time = time.time() - start_time

# model.save("./_save/keras23_7_save_model2.h5")
# model.save_weights("./_save/keras23_7_save_weights2.h5")

model = load_model("./_save/keras23_7_save_model2.h5")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  


#===================== save_model1 (random 모델) =================================#
# loss :  [2.2956061363220215, 11.287386894226074]
# r2 스코어:  0.8633771105613022
#=================================================================================#

#===================== save_model2 (훈련한 모델) ==================================#
# loss :  [2.3675475120544434, 12.978792190551758]
# r2 스코어:  0.8429042899619104
#=================================================================================#
#===================== save_weights1 (random 한 가중치 값) ========================#
# loss :  [22.001361846923828, 566.6704711914062]
# r2 스코어:  -5.85899779288622
#=================================================================================#

#===================== save_weights2 (훈련한 가중치 값) ===========================#
# loss :  [2.3675475120544434, 12.978792190551758]
# r2 스코어:  0.8429042899619104
#=================================================================================#