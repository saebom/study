from attr import validate
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split, validation_curve

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))
# [실습] train_test_split만 사용
# 10:3:3 으로 나누기

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.18, random_state=1
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.18, random_state=1)

print(x_train, y_train, x_test, y_test, x_val, y_val)

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