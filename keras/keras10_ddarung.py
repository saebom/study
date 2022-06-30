# 데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)

print(train_set)
print(train_set.shape)  # (1459, 10)

test_set = pd.read_csv(path + 'test.csv',   # 예측에서 사용!!
                       index_col=0)

print(test_set)
print(test_set.shape)   # (715, 9)  => count 제외한 9개 :: 결과적으로 count 값을 제출

print(train_set.columns)
print(train_set.info())
print(train_set.describe()) #pandas api 에서는 info(), describe() 제공 

#### 결측치 처리 ####
print(train_set.isnull().sum()) # null의 합계 출력
# trian_set = train_set.replace(np.nan, 0, inplace=True)  # nan 대신에 0을 넣음
# train_set = train_set.dropna()  # nan 값 삭제
test_set = test_set.fillna(method = 'ffill')
train_set = train_set.fillna(method = 'ffill')
print(train_set.isnull().sum()) # null의 합계 출력
print(train_set.shape)  # (1328, 10)


x = train_set.drop(['count'], axis=1)

print(x)
print(x.columns)
# print(x.shape)  # (1459, 9)

y = train_set['count']
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=31
)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=9))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=5000, batch_size=100)

#4. 평가예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)  

#### mse를 rmse로 변환 ####
y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# loss :  2020.923583984375
# RMSE :  44.95468636343962
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.9, shuffle=True, random_state=31
# )
# model = Sequential()
# model.add(Dense(100, input_dim=9))
# model.add(Dense(100))
# model.add(Dense(1))
# epochs=500, batch_size=100

# 데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)

print(train_set)
print(train_set.shape)  # (1459, 10)

test_set = pd.read_csv(path + 'test.csv',   # 예측에서 사용!!
                       index_col=0)

print(test_set)
print(test_set.shape)   # (715, 9)  => count 제외한 9개 :: 결과적으로 count 값을 제출

print(train_set.columns)
print(train_set.info())
print(train_set.describe()) #pandas api 에서는 info(), describe() 제공 

#### 결측치 처리 ####
print(train_set.isnull().sum()) # null의 합계 출력
# trian_set = train_set.replace(np.nan, 0, inplace=True)  # nan 대신에 0을 넣음
train_set = train_set.dropna()  # nan 값 삭제
print(train_set.isnull().sum()) # null의 합계 출력
print(train_set.shape)  # (1328, 10)


x = train_set.drop(['count'], axis=1)

print(x)
print(x.columns)
# print(x.shape)  # (1459, 9)

y = train_set['count']
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=1004
)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=9))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(400))
model.add(Dense(500))
model.add(Dense(700))
model.add(Dense(500))
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
# model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=110)

#4. 평가예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)  # loss :  2775.9921875

#### mse를 rmse로 변환 ####
y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# loss :  2430.768310546875
# RMSE :  49.3028218000355
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.9, shuffle=True, random_state=1004
# )
# epochs=500, batch_size=110

