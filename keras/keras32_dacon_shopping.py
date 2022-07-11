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
import seaborn as sns
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

# 결측치 처리 
# Promotion NaN값 변환
train_set['Promotion1'] = train_set['Promotion1'].fillna(train_set['Promotion1'].mode()[0])
train_set['Promotion2'] = train_set['Promotion2'].fillna(train_set['Promotion2'].mode()[0])
train_set['Promotion3'] = train_set['Promotion3'].fillna(train_set['Promotion3'].mode()[0])
train_set['Promotion4'] = train_set['Promotion4'].fillna(train_set['Promotion4'].mode()[0])
train_set['Promotion5'] = train_set['Promotion5'].fillna(train_set['Promotion5'].mode()[0])

test_set['Promotion1'] = test_set['Promotion1'].fillna(test_set['Promotion1'].mode()[0])
test_set['Promotion2'] = test_set['Promotion2'].fillna(test_set['Promotion2'].mode()[0])
test_set['Promotion3'] = test_set['Promotion3'].fillna(test_set['Promotion3'].mode()[0])
test_set['Promotion4'] = test_set['Promotion4'].fillna(test_set['Promotion4'].mode()[0])
test_set['Promotion5'] = test_set['Promotion5'].fillna(train_set['Promotion5'].mode()[0])

# Date 칼럼에서 "월"에 해당하는 정보만 추출하여 숫자 형태로 반환
def get_month(date):
    month = date[3:5]
    month = int(month)
    return month

# 이 함수를 Date 칼럼에 적용한 Month 칼럼을 만들어줍니다.
train_set['Month'] = train_set['Date'].apply(get_month)
test_set['Month'] = test_set['Date'].apply(get_month)

# 결과를 확인합니다.
print(train_set)

# Date의 object 데이터를 int64 데이터로 변환 및 결측치 제거
cat_col = train_set.dtypes[train_set.dtypes == 'object'].index   
for col in cat_col:
    train_set[col] = train_set[col].fillna('None')
    train_set[col] = LabelEncoder().fit_transform(train_set[col].values)
print(train_set.head(5))     # ==> 2줄까지 출력하여 확인  
  
cat_col = test_set.dtypes[test_set.dtypes == 'object'].index   
for col in cat_col:
    test_set[col] = test_set[col].fillna('None')
    test_set[col] = LabelEncoder().fit_transform(test_set[col].values)
       
train_set['IsHoliday'] = train_set['IsHoliday'].fillna(train_set.IsHoliday.dropna().mode()[0])
train_set['IsHoliday'] = train_set['IsHoliday'].apply(np.round).astype('float64')
test_set['IsHoliday'] = test_set['IsHoliday'].fillna(train_set.IsHoliday.dropna().mode()[0])
test_set['IsHoliday'] = test_set['IsHoliday'].apply(np.round).astype('float64')

print(train_set.shape, test_set.shape)      # (6255, 13) (180, 12)                                                  
print(train_set.head(), test_set.head())
        
train_set = train_set.drop(columns=['id', 'Date'])
test_set = test_set.drop(columns=['id', 'Date'])
        
x = train_set.drop(['Weekly_Sales'], axis=1)
print(x)
print(x.columns)
# print(x.shape)  

y = train_set['Weekly_Sales']
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=31
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='linear', input_dim=11))
model.add(Dropout(0.2))     
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))     
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))     
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
hist = model.fit(x_train, y_train, epochs=500, batch_size=32,
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
y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape)   # (1458, 1)

submission = pd.read_csv(path + 'submission.csv')
submission['Weekly_Sales'] = y_summit

# Brutal approach to deal with predictions close to outer range 
q1 = submission['Weekly_Sales'].quantile(0.0045)
q2 = submission['Weekly_Sales'].quantile(0.99)

submission['Weekly_Sales'] = submission['Weekly_Sales'].apply(lambda x: x if x > q1 else x*0.77)
submission['Weekly_Sales'] = submission['Weekly_Sales'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv(path + 'submission1.csv', index=False)


#4_1. 그래프로 비교
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['Weely_Sales'], marker='.', c='red', label='loss')
# plt.plot(hist.history['Promotion1'], marker='.', c='blue', label='val_loss')
# plt.grid()
# # plt.title('loss & val_loss')    
# plt.title('Weekly_Sales & Promotion1')    
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend()   # 자동으로 빈 공간에 라벨표시
# plt.show()


#4_2. loss & val_loss
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
# loss :  [412308.9375, 271772106752.0]
# RMSE :  521317.6636425292
# R2 :  0.16262125708309716
#===================================================================================#
