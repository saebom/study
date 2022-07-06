import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
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

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation = 'linear', input_dim=9))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss = 'mae', optimizer="adam", metrics=['mse'])

earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='min',
                              restore_best_weights=True,
                              verbose=1)
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

y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape)   # (715, 1)

# submission = pd.read_csv(path + 'submission.csv')
# submission['count'] = y_summit
# submission.to_csv(path + 'submission1.csv', index=False)


# print("=================================================================")   
# print(hist)     # <tensorflow.python.keras.callbacks.History object at 0x000002664FF27AF0>
# print("=================================================================")
# print(hist.history)     # loss 와 val_loss의 key, value를 합쳐놓은 것
# print("=================================================================")
# print(hist.history['loss'])
# print("=================================================================")
# print(hist.history['val_loss'])


# 그래프로 비교
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


#============================ Scaler 적용 전 데이터 ================================#
# loss : 25.755203247070312
# mse :  1323.28759765625
# r2 스코어:  0.7539495137131228
#==================================================================================#

#=========================== MinMaxScaler 적용 후 데이터 ===========================#
# loss : 28.491697311401367
# mse :  1769.7752685546875
# r2 스코어:  0.7152692166754488
#==================================================================================#

#=========================== StandardScaler 적용 후 데이터 =========================#
# loss : 30.303726196289062
# mse :  1848.76708984375
# r2 스코어: 0.7025605916612034
#==================================================================================#

#=========================== MaxAbsScaler 적용 후 데이터 ===========================#
# loss : 29.747373580932617
# mse :  1939.935302734375
# r2 스코어: 0.6878929301761594
#==================================================================================#

#=========================== RobustScaler 적용 후 데이터 ===========================#
# loss : 31.266366958618164
# mse :  2060.465576171875
# r2 스코어: 0.6685014327912705
#==================================================================================#
