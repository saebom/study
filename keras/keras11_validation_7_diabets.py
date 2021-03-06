import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

datasets = load_diabetes()
x = datasets.data
y = datasets.target

'''
print(x)
print(y)

print(x.shape, y.shape)   # (442, 10) (442,)
print(datasets.feature_names)
print(datasets.DESCR)
'''
# [실습] train_size 조절 가능, R2 0.62 이상

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=72
)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(248))
model.add(Dense(124))
model.add(Dense(62))
model.add(Dense(7))
model.add(Dense(1))


#3. 훈련
model.compile(loss='mse', optimizer='adam')   
# model.fit(x_train, y_train, epochs=200, batch_size=20)
model.fit(x_train, y_train, epochs=200, batch_size=20, validation_split=0.15)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  

r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  

#==================================#
# loss :  2181.05029296875
# r2 스코어:  0.6696989220664229

# => validation 검증 후 
# loss :  2116.52099609375
# r2 스코어:  0.6337916880044802
#==================================#

