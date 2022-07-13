import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sqlalchemy import column
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from scipy.stats import skew
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
path = './_data/shopping/'
train_set = pd.read_csv(path + 'train.csv')
print(train_set)
print(train_set.shape)   # (6255, 13)
print(train_set.columns) # ['id', 'Store', 'Date', 'Temperature', 'Fuel_Price', 'Promotion1',
                         # 'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'Unemployment',
                         # 'IsHoliday', 'Weekly_Sales']
                        
print(train_set.info())
print(train_set.describe())
print(train_set.isnull().sum())  # 'Promotion1', 'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5'


test_set = pd.read_csv(path + 'test.csv')
print(test_set)
print(test_set.shape)   # (180, 12)
print(test_set.columns) # ['id', 'Store', 'Date', 'Temperature', 'Fuel_Price', 'Promotion1',
                        #  'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'Unemployment',
                        #  'IsHoliday']   ==> 'Weekly_Sales'
print(test_set.info())
print(test_set.describe())
print(test_set.isnull().sum())  # 'Promotion1', 'Promotion2', 'Promotion3', 'Promotion4'

###### 데이터 전처리 ######

# Date 날짜 변수 분리 
# train_set['Date'] = pd.to_datetime(train_set['Date'])
# train_set['year'] = train_set['Date'].dt.year
# train_set['month'] = train_set['Date'].dt.month
# train_set['week'] = train_set['Date'].dt.isocalendar().week
# train_set['quarter'] = train_set['Date'].dt.quarter
# train_set['day_of_week'] = train_set['Date'].dt.day_name()
# print(train_set.head(5))


#  Weekly_Sales 그래프 확인
# plt.hist(train_set.Weekly_Sales, bins=50)
# plt.show()

# 매장별 주간매출액
pd.options.display.float_format = '{:.2f}'.format
dm = train_set.groupby('Store')['Weekly_Sales'].agg(**{'mean_Weekly_Sales':'mean'}).sort_values('mean_Weekly_Sales', ascending=False)
print(dm)

# 지점별 상관관계 분석
from matplotlib import dates
import seaborn as sns
# fig = plt.figure(figsize=(35,35))
# fig.set_facecolor('white')

# for i in range(35, 36):
#     train1 = train_set[train_set.Store == i] 
#     plt.figure(figsize=(18, 18))
#     sns.heatmap(data=train1.corr(), annot=True)
# plt.show()

# Weekly_Sales 제거
label = train_set['Weekly_Sales']
all_data_set = pd.concat((train_set, test_set)).reset_index(drop=True)
all_data_set = all_data_set.drop(['Weekly_Sales'], axis=1)
print(all_data_set.shape)   
print(all_data_set) # (6435, 12)

# Date 날짜 변수 분리 
all_data_set['Date'] = pd.to_datetime(all_data_set['Date'])
all_data_set['year'] = all_data_set['Date'].dt.year
all_data_set['month'] = all_data_set['Date'].dt.month
all_data_set['week'] = all_data_set['Date'].dt.isocalendar().week
all_data_set['quarter'] = all_data_set['Date'].dt.quarter
all_data_set['day_of_week'] = all_data_set['Date'].dt.day_name()
print(all_data_set.head(5))


fig = plt.figure(figsize=(35,35))
fig.set_facecolor('white')

for i in range(35, 36):
    all_data1 = all_data_set[all_data_set.Store == i] 
    plt.figure(figsize=(18, 18))
    sns.heatmap(data=all_data1.corr(), annot=True)
plt.show()


# Date의 object 데이터를 int64 데이터로 변환 및 결측치 제거
cat_col = all_data_set.dtypes[all_data_set.dtypes == 'object'].index   
for col in cat_col:
    all_data_set[col] = all_data_set[col].fillna('None')
    all_data_set[col] = LabelEncoder().fit_transform(all_data_set[col].values)
print(all_data_set.head(5))     # ==> 5줄까지 출력하여 확인  
  
all_data_set['IsHoliday'] = all_data_set['IsHoliday'].fillna(all_data_set.IsHoliday.dropna().mode()[0])
all_data_set['IsHoliday'] = all_data_set['IsHoliday'].apply(np.round).astype('float64')

print(all_data_set.shape)      # (6255, 13) (180, 12)                                                  
print(all_data_set.head())


# feature 제거        
all_data_set = all_data_set.drop(columns=['id', 'Date'])
all_data_set = all_data_set.drop(columns=['Fuel_Price', 'Promotion1', 'Promotion2', 'Promotion4', 'year'])
print(all_data_set.head())

# 데이터 확인
print(all_data_set.isnull().sum())

# 결측치 처리 
# Promotion NaN값 변환
all_data_set['Promotion3'] = all_data_set['Promotion3'].fillna(all_data_set['Promotion3'].median())
all_data_set['Promotion5'] = all_data_set['Promotion5'].fillna(all_data_set['Promotion5'].median())
print(all_data_set.head(5))
print(all_data_set.isnull().sum())
 
# get_dummies
all_data = pd.get_dummies(all_data_set)
print(all_data_set.shape)   # (6435, 10)

######## Model libraries import ########
# all_data_set을 train_set과 test_set으로 분할
train_set = all_data_set[:len(train_set)]
test_set = all_data_set[len(test_set):]
print(train_set.shape, test_set.shape)  # (6255, 15) (180, 15)

x = train_set
y = label

print(x.shape, y.shape) # (6255, 15) (6255,)

# train 데이터와 test 데이터의 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9713, random_state =31)

# train 데이터와 test 데이터의 분리 결과 확인
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (6075, 10) (180, 10) (6075,) (180,)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='linear', input_dim=10))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# 3. 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mse'])  

import datetime
date = datetime.datetime.now()     
date = date.strftime("%m%d_%H%M")   
print(date)

filepath = './_ModelCheckPoint/k32/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '01_', date, '_', filename])
                      )
start_time = time.time() 
hist = model.fit(x_train, y_train, epochs=1000, batch_size=128,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)  
      
#### mse를 rmse로 변환 ####
y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
print(y_predict[:10])

# r2 값 구하기  
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)  

#### summit ####
y_summit = model.predict(x_test)
print(y_summit)
print(y_summit.shape)   # (180, 15)

submission = pd.read_csv(path + 'submission.csv')
submission['Weekly_Sales'] = y_summit
submission.to_csv(path + 'submission1.csv', index=False)


#4_1. 그래프로 비교
font_path = 'C:\Windows\Fonts\malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
# plt.title('loss & val_loss')    
plt.title('로스값과 검증로스값')    
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right')   # 우측상단에 라벨표시
plt.legend()   # 자동으로 빈 공간에 라벨표시
plt.show()


#====================================== 결과1 =======================================#
# loss :  [410222.375, 292435918848.0]
# RMSE :  540773.4482686959
# R2 :  0.09895232346019256
#===================================================================================#


#====================================== 결과2 =======================================#
# loss :  [204932.28125, 107756879872.0]
# RMSE :  328263.4366404997
# R2 : 0.6418582901643641
#===================================================================================#
