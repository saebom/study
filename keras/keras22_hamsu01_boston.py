import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model
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

# # 함수형 모델
# input1 = Input(shape=(13,))
# dense1 = Dense(7)(input1)
# dense2 = Dense(10, activation='linear')(dense1)
# dense3 = Dense(30, activation='relu')(dense2)
# dense4 = Dense(40, activation='relu')(dense3)
# dense5 = Dense(50, activation='relu')(dense4)
# dense6 = Dense(100, activation='relu')(dense5)
# dense7 = Dense(50, activation='relu')(dense6)
# dense8 = Dense(40, activation='relu')(dense7)
# dense9 = Dense(30, activation='relu')(dense8)
# dense10 = Dense(10, activation='relu')(dense9)
# dense11 = Dense(7, activation='linear')(dense10)
# output1 = Dense(1, activation='linear)(dense11)
# model = Model(inputs=input1, outputs=output1)


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
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  

print("=================================================================")
print("걸린시간 : ", end_time)

# print("=================================================================")   
# print(hist)    
# print("=================================================================")
# print(hist.history)     # loss 와 val_loss의 key, value를 합쳐놓은 것
# print("=================================================================")
# print(hist.history['loss'])
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


#================================== Sequential 모델 ================================#
# 걸린시간 :  57.09792113304138
# loss : 2.2751407623291016
# mse :  10.479668617248535
# r2 스코어:  0.8731537585400484
#==================================================================================#

#==================================== 함수형 모델 ===================================#
# 걸린시간 :  43.46778655052185
# loss :  2.2860517501831055
# mse :  12.258877754211426
# r2 스코어:  0.8516181421237415
#==================================================================================#
