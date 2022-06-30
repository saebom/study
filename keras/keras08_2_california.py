import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


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
    x, y, train_size=0.7, shuffle=True, random_state=1004
)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))


#3. 훈련
model.compile(loss='mse', optimizer='adam')   
model.fit(x_train, y_train, epochs=5000, batch_size=200)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  

r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  

# loss :  0.5457625389099121
# r2 스코어:  0.5926412793973783
