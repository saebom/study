import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=8))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))


#3. 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mse'])   

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=50, mode='min', verbose=1, 
                              restore_best_weights=True)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=100, 
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)      # 0.7170546650886536

y_predict = model.predict(x_test)  
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  


# print("=================================================================")   
# print(hist)     
# print("=================================================================")
# print(hist.history)     
print("=================================================================")
print(hist.history['loss'])
print("=================================================================")
# print(hist.history['val_loss'])

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = 'C:\Windows\Fonts\malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
# plt.title('loss & val_loss')    
plt.title('로스값과 검증로스값')   
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right')  
plt.legend()   
plt.show()


#============================ Scaler 적용 전 데이터 ===============================#
# loss :  0.4930032193660736
# mse :  0.4930032193660736 
# r2 스코어:  0.6412728093980673
#==================================================================================#

#=========================== MinMaxScaler 적용 후 데이터 ============================#
# loss :  0.2561401128768921
# mse :  0.2561401128768921
# r2 스코어:   0.8133321552071293
#==================================================================================#

#=========================== StandardScaler 적용 후 데이터 ==========================#
# loss :  0.27067136764526367
# mse :  0.27067136764526367
# r2 스코어:  0.8027422202156695
#==================================================================================#

#=========================== MaxAbsScaler 적용 후 데이터 ==========================#
# loss :  0.33722156286239624
# mse :  0.33722156286239624
# r2 스코어:  0.7542421946329673
#==================================================================================#

#=========================== RobustScaler 적용 후 데이터 ==========================#
# loss :  0.2945418953895569
# mse :  0.2945418953895569
# r2 스코어:  0.7853458715414465
#==================================================================================#
