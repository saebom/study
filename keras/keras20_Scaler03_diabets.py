import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(datasets.DESCR)

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
model.add(Dense(100, activation='linear', input_dim=10))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', 
                              verbose=1, 
                              restore_best_weights=True)
start_time = time.time() 
hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  


# print("=================================================================")   
# print(hist)     # <tensorflow.python.keras.callbacks.History object at 0x000002664FF27AF0>
# print("=================================================================")
# print(hist.history)     # loss 와 val_loss의 key, value를 합쳐놓은 것
# print("=================================================================")
# print(hist.history['loss'])
# print("=================================================================")
# print(hist.history['val_loss'])


#4_1. 그래프로 비교
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
# plt.legend(loc='upper right')   # 우측상단에 라벨표시
plt.legend()   # 자동으로 빈 공간에 라벨표시
plt.show()

#============================ Scaler 적용 전 데이터 ===============================#
# loss :  2191.158447265625
# mse :  2191.158447265625
# r2 스코어:  0.6681681298699478
#==================================================================================#

#=========================== MinMaxScaler 적용 후 데이터 ============================#
# loss :  3081.5048828125
# mse :  3081.5048828125
# r2 스코어:   0.5054058286481318
#==================================================================================#

#=========================== StandardScaler 적용 후 데이터 ==========================#
# loss :  3299.023193359375
# mse :  3299.023193359375
# r2 스코어:  0.4704932330219411
#==================================================================================#

#=========================== MaxAbsScaler 적용 후 데이터 ==========================#
# loss :  3501.8212890625
# mse :  3501.8212890625
# r2 스코어:  0.4379433736122881
#==================================================================================#

#=========================== RobustScaler 적용 후 데이터 ==========================#
# loss :  3300.675048828125
# mse :  3300.675048828125
# r2 스코어:  0.47022815940083595
#==================================================================================#