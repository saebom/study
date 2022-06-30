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

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=66
)
print(x_test.shape, x_train.shape)      # (152, 13) (354, 13)


#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))


#3. 훈련
model.compile(loss='mse', optimizer='adam')   
model.fit(x_train, y_train, epochs=200, batch_size=10,
          validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  

r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  


#====================================#
# loss :  22.993427276611328
# r2 스코어:  0.759708421864608

# ==> validation 검증 후 
# loss :  33.65226364135742
# r2 스코어:  0.6483188364372215
#====================================#