import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
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
datasets = load_boston()
x = datasets.data
y = datasets['target']

# print(np.min(x))    # 0.0
# print(np.max(x))    # 711.0
# x = (x - np.min(x)) / (np.max(x)-np.min(x))
# print(x[:10])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(np.min(x_train))    # 0.0
# print(np.max(x_train))    # 1.0000000000000002
# print(np.min(x_test))    # 0.0
# print(np.max(x_test))

# a = 0.1
# b = 0.2
# print(a+b)  # 0.30000000000000004

# 보스턴에 대해서 3가지 비교
# 1. 스케일러 하기 전
# 2. 민맥스
# 3. 스탠다드 => 3가지 성능 비교

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


#3. 훈련
model.compile(loss='mae', optimizer='adam',
              metrics=['mse']) 
earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True)  

start_time =time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, 
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1) 
end_time =time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  

# print("=================================================================")   
# print(hist)    
print("=================================================================")
print(hist.history)     # loss 와 val_loss의 key, value를 합쳐놓은 것
print("=================================================================")
print(hist.history['loss'])
# print("=================================================================")
# print(hist.history['val_loss'])


#4-1. 시각화
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
plt.legend()  
plt.show()

#============================ Scaler 적용 전 데이터 ===============================#
# loss :  2.4444220066070557
# mse :  11.027112007141113 
# r2 스코어:  0.8680698301243783
#==================================================================================#

#=========================== MinMaxScaler 적용 후 데이터 ============================#
# loss :  2.467968702316284
# mse :  10.638111114501953
# r2 스코어:   0.8712359745153038
#==================================================================================#

#=========================== StandardScaler 적용 후 데이터 ==========================#
# loss : 2.208564519882202
# mse :  11.264325141906738
# r2 스코어:  0.8636562521620402
#==================================================================================#

#=========================== MaxAbsScaler 적용 후 데이터 ==========================#
# loss : 2.2751407623291016
# mse :  10.479668617248535
# r2 스코어:  0.8731537585400484
#==================================================================================#

#=========================== RobustScaler 적용 후 데이터 ==========================#
# loss : 2.303745985031128
# mse :  10.724520683288574
# r2 스코어:   0.8701900527179431
#==================================================================================#
