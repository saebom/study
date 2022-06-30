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
print(test_set.isnull().sum()) # null의 합계 출력
# trian_set = train_set.replace(np.nan, 0, inplace=True)  # nan 대신에 0을 넣음
# test_set = test_set.fillna(test_set.mean())
test_set = test_set.fillna(method='ffill')
# test_set = test_set.fillna({'hour_bef_temperature':int(test_set['hour_bef_temperature'].mean())}, inplace=True)
# test_set = test_set.fillna({'hour_bef_precipitation':int(test_set['hour_bef_precipitation'].mean())}, inplace=True)
# test_set = test_set.fillna({'hour_bef_windspeed':int(test_set['hour_bef_windspeed'].mean())}, inplace=True)
# test_set = test_set.fillna({'hour_bef_humidity':int(test_set['hour_bef_humidity'].mean())}, inplace=True)
# test_set = test_set.fillna({'hour_bef_visibility':int(test_set['hour_bef_visibility'].mean())}, inplace=True)
# test_set = test_set.fillna({'hour_bef_ozone':int(test_set['hour_bef_ozone'].mean())}, inplace=True)
# test_set = test_set.fillna({'hour_bef_pm10':int(test_set['hour_bef_pm10'].mean())}, inplace=True)
# test_set = test_set.fillna({'hour_bef_pm2.5':int(test_set['hour_bef_pm2.5'].mean())}, inplace=True)
# test_set = test_set.replace(np.nan, {'hour_bef_temperature':np.float64(test_set['hour_bef_temperature'].mean())}, inplace=True)
# test_set = test_set.replace(np.nan, {'hour_bef_precipitation':np.float64(test_set['hour_bef_precipitation'].mean())}, inplace=True)
# test_set = test_set.replace(np.nan, {'hour_bef_windspeed':np.float64(test_set['hour_bef_windspeed'].mean())}, inplace=True)
# test_set = test_set.replace(np.nan, {'hour_bef_humidity':np.float64(test_set['hour_bef_humidity'].mean())}, inplace=True)
# test_set = test_set.replace(np.nan, {'hour_bef_visibility':np.float64(test_set['hour_bef_visibility'].mean())}, inplace=True)
# test_set = test_set.replace(np.nan, {'hour_bef_ozone':np.float64(test_set['hour_bef_ozone'].mean())}, inplace=True)
# test_set = test_set.replace(np.nan, {'hour_bef_pm10':np.float64t(test_set['hour_bef_pm10'].mean())}, inplace=True)
# test_set = test_set.replace(np.nan, {'hour_bef_pm2.5':np.float64t(test_set['hour_bef_pm2.5'].mean())}, inplace=True)
# train_set = train_set.dropna()  # nan 값 삭제
train_set = train_set.fillna(method='ffill')
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
model.add(Dense(100, activation='elu'))
model.add(Dense(100, activation='elu'))
model.add(Dense(100, activation='elu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer="nadam")
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

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


# test_set = test_set.replace(np.nan, 0, inplace=True)  # nan 대신에 0을 넣음
y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape)   # (715, 1)

####### .to_csv() 를 사용해서 
#### submission.csv를 완성하시오 !!! 

submission = pd.read_csv('./_data/ddarung/submission.csv')
submission['count'] = y_summit
print(submission)
submission.to_csv('./_data/ddarung/submission3.csv', index=False)


# loss :  12.729743003845215
# RMSE :  3.5678786869767625
# R2 :  0.9976330443981015


# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.9, shuffle=True, random_state=31
# )
# model = Sequential()
# model.add(Dense(100, input_dim=9))
# model.add(Dense(100, activation='elu'))
# model.add(Dense(100, activation='elu'))
# model.add(Dense(100, activation='elu'))
# model.add(Dense(1))
# model.compile(loss = 'mse', optimizer="nadam")
# model.fit(x, y, epochs=5000, batch_size=100)
