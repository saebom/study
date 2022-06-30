# https://www.kaggle.com/competitions/bike-sharing-demand/overview

# Kaggle Bike Sharing Demand 문제풀이


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
path = './_data/bike/'
train_set = pd.read_csv(path + 'train.csv')
print(train_set)
print(train_set.shape)   # (10886, 11)
print(train_set.columns) # ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                         #  'humidity', 'windspeed', 'casual', 'registered', 'count']
                        
# print(train_set.info())
# print(train_set.describe())
# print(train_set.isnull().sum())

test_set = pd.read_csv(path + 'test.csv')
print(test_set)
print(test_set.shape)   # (6493, 8)
print(test_set.columns) # ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                        #  'humidity', 'windspeed']   ==> casual, registered, count
# print(test_set.info())
# print(test_set.describe())
# print(test_set.isnull().sum())  


##### 결측치 제거 ######

train_set = train_set[train_set['weather'] != 4]  # weather가 4인 데이터는 이상치(폭우, 폭설이 내리는 날 저녁 6시에 대여)


# object ==> 날짜 형식으로 변환
# from datetime import datetime
from pandas import DataFrame


'''
for df in (train_set, test_set):
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['minute'] = df['datetime'].dt.second
    
test_set.head()
'''

train_set['datetime'] = pd.to_datetime(train_set['datetime'])
test_set['datetime'] = pd.to_datetime(test_set['datetime'])

train_set['year'] = train_set['datetime'].dt.year
train_set['month'] = train_set['datetime'].dt.month
train_set['day'] = train_set['datetime'].dt.day
train_set['hour'] = train_set['datetime'].dt.hour
train_set['minute'] = train_set['datetime'].dt.minute
train_set['second'] = train_set['datetime'].dt.second

test_set["year"] = test_set["datetime"].dt.year
test_set["month"] = test_set["datetime"].dt.month
test_set["day"] = test_set["datetime"].dt.day
test_set["hour"] = test_set["datetime"].dt.hour
test_set["minute"] = test_set["datetime"].dt.minute
test_set["second"] = test_set["datetime"].dt.second


#데이터 합치기
all_data_set = pd.concat([train_set, test_set])     
all_data_set
print('all_data_set :', all_data_set)

# all_data_set = pd.concat([train_set, test_set], ignore_index=True)
# all_data_set
# print('all_data_set값 : ', all_data_set)


# 필요없는 feature 제거
drop_all_data_set = ['casual', 'registered', 'datetime', 'date', 'windspeed', 'month']
drop_all_data_set = ['casual', 'registered']
all_data_set = all_data_set.drop(drop_all_data_set, axis=1)
print('all_data값(2) : ', all_data_set)

# all_data_set = pd.DataFrame({'datetime'})
# all_data_set
# print('errorrrrrrrrrrrr :', all_data_set)  
# print(all_data_set.shape)

'''
# 날짜 feature 생성
all_data_set['date'] = all_data_set[datetime].apply(lambda x:x.split()[0])

# 연도 feature 생성
all_data_set['year'] = all_data_set[datetime].apply(lambda x:x.split()[0].split('-')[0])

# 월 feature 생성
all_data_set['month'] = all_data_set[datetime].apply(lambda x:x.split()[0].split('-')[1])

# 시 feature 생성
all_data_set['hour'] = all_data_set[datetime].apply(lambda x:x.split()[1].split(':')[0])

# 요일 feature 생성
all_data_set['weekday'] = all_data_set['date'].apply(lambda dateString: datetime.strptime(dateString, '%Y-%m-%d').weekday())
'''
# 훈련데이터와 테스트데이터 나누기
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.9, shuffle=True, random_state=66
# )
x_train = all_data_set[~pd.isnull(all_data_set['count'])]   # count가 있으면 훈련데이터
x_test = all_data_set[~pd.isnull(all_data_set['count'])]    # count가 없으면 테스트데이터

# count 제거
x_train = x_train.drop(['count'], axis=1)
x_test = x_test.drop(['count'], axis=1)

y = train_set['count']

print(y)
print(y.shape)  # (10886,)

# feature 엔지니어링 후의 훈련 데이터
x_train.head()

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y, epochs=500, batch_size=100)

#4. 평가예측
loss = model.evaluate(x_test, y)
print('loss : ', loss)  

#### mse를 rmse로 변환 ####
y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y, y_predict)
print("RMSE : ", rmse)

#### summit #### 
y_summit = model.predict(test_set)
# y_summit = abs(y_summit)
print(y_summit)
print(y_summit.shape)  

submission = pd.read_csv(path + 'sampleSubmission.csv')
submission['count'] = y_summit
submission.to_csv('./_data/bike/submission.csv', index=True)
