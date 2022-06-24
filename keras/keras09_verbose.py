from tabnanny import verbose
import numpy as ny
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

'''
print(x)
print(y)   # y는 보스턴 집 값
print(x.shape, y.shape)   # (506, 13), (506,)

print(datasets.feature_names)    # feature_name은 사이킷런에서 예제용으로 제공된 데이터의 이름 
print(datasets.DESCR)
'''

#[실습] 아래를 완성할 것
#1) train 0.7
#2) R2를 0.8 이상

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=66
)

#2. 모델 구성
model = Sequential()
model.add(Dense(7, input_dim=13))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(1))

import time

#3. 훈련
model.compile(loss='mse', optimizer='adam')  
start_time = time.time()      #1656032693.2072284
print(start_time) 
model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=0)  
end_time = time.time() - start_time

print("걸린시간 : ", end_time)

'''
verbose 0 걸린시간 : 2.2564609050750732 / 출력없음
verbose 1 걸린시간 : 2.401794672012329  / 잔소리 많음
verbose 2 걸린시간 : 2.2612545490264893 / 프로그래스바 없음
verbose 3 걸린시간 : 2.1730453968048096  / epoch만 나옴 
verbose 4 걸린시간 : 2.201508045196533  / epoch 횟수만 나옴 

'''