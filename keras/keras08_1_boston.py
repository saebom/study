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
    x, y, train_size=0.7, shuffle=True, random_state=72
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
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(1))


#3. 훈련
model.compile(loss='mse', optimizer='adam')   
model.fit(x_train, y_train, epochs=200, batch_size=10)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  

r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  


#===================================== 결  과 ==========================================#
# 결과 acc :  0.6286952104589564
# r2 스코어:  0.6286952104589564
#=======================================================================================#