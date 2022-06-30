from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))
# [실습] slicing 해서 만들기

# x_train = x[:10]
# y_train = y[:10]
# x_test = x[10:13]
# y_test = y[10:13]
# x_val = x[13:17]
# y_val = y[13:17]
# print(x_train, y_train, x_test, y_test, x_val, y_val)
 
x_train = np.array(range(1, 11))    # 0~10
y_train = np.array(range(1, 11))    # 0~10
x_test = np.array([11, 12, 13])     # 11~13, test는 evaluate와 predict 에서 사용함
y_test = np.array([11, 12, 13])     # 11~13
x_val = np.array([14, 15, 16])      # 14~16
y_val = np.array([14, 15, 16])      # 14~16

# #2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값 : ", result)

#### 통상적으로 val의 loss값이 loss값보다 좋지 않음
# 검증데이터는 통상 일반적 훈련 때보다 좋지 않음