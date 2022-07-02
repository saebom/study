# 과제
# activation : sigmoid, relu, linear 넣어라
# metrics 추가
# EarlyStopping 넣구
# 성능비교
# 느낀점 2줄 이상!!!

from gc import callbacks
from lightgbm import early_stopping
import numpy as ny
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=66
)

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
              metrics=['mse'])  # 회귀모델의 경우 loss='mae', metrics='mse' 사용 (회귀모델이므로 accuracy=0)

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True)  

start_time =time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, 
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1) 
end_time =time.time()


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
# plt.legend(loc='upper right')   # 우측상단에 라벨표시
plt.legend()   # 자동으로 빈 공간에 라벨표시
plt.show()


#================================ 적용 전 데이터 ===================================#
# loss :  28.372167587280273
# r2 스코어:  0.6440093136844579
#==================================================================================#

#================================ 적용 후 데이터 ===================================#
# loss :  2.4444220066070557
# mse :  11.027112007141113 
# r2 스코어:  0.8680698301243783
#==================================================================================#

##=============================== 내 용 정 리 ======================================##
# 1) 회귀모델이므로 loss='mae', metrics=['mse'] 사용
#   (회귀모델에서는 'accuracy'= 0이 나옴)
# 2) 회귀모델이므로 accuracy_score 가 아니라 r2 값 사용
# 주의!!! output 레이어에 'sigmoid'가 아니라 'linear' 사용해야 함(분류모델 아니까!!!)
##=================================================================================##
         
##================================ 느 낀 점 ========================================##
# 처음에 output 레이어에 'sigmoid'가 아니라 'linear' 적용은 잊지않고 하였으나, 
# loss = 'binary_crossentropy'로 적용했더니 훈련 증 loss 값에 NaN이 나와서 당황하였음
# 회귀모델은 loss = 'mae', metrics=['mse']로 사용해야한다는 것을 알게됨
##=================================================================================##
         
