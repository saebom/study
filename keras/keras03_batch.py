import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
 
# 1. 데이터
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])
 
# 2. 모델 구성
model = Sequential()
model.add(Dense(7, input_dim=1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련, batch
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측 
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])
print('6의 예측값 : ', result)

# loss :  0.41068750619888306
# 6의 예측값 :  [[6.0084367]]