import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

'''
print(x)
print(y)
print(x.shape, y.shape)  # (20640, 8) (206040,)

print(datasets.feature_names)
print(datasets.DESCR)
'''

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=0
)

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



#================================ 적용 전 데이터 ===================================#
# loss : 0.662193238735199 
# r2 스코어: 0.4921656465420844
#==================================================================================#

#================================ 적용 후 데이터 ===================================#
# loss :  0.4930032193660736
# mse :  0.4930032193660736 
# r2 스코어:  0.6412728093980673
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
         
