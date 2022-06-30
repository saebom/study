import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [실습] train과 test를 섞어서 7:3으로 찾을 수 있는 방법을 찾아라  => 섞음(shuffle), 무작위(random_state)
from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=1004)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, 
    train_size=0.7, 
    # shuffle=True, 
    random_state=66
)

print(x_train) # [2 7 6 3 4 8 5]
print(x_test)  # [ 1  9 10]
print(y_train) # [2 7 6 3 4 8 5]
print(y_test)  # [ 1  9 10]


#2. 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

#3. 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 700, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('[11]의 예측값 : ', result)

# loss :  4.6199488679121714e-11
# [11]의 예측값 :  [[10.99999]]
