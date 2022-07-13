import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sqlalchemy import asc, column
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
fig = plt.figure(figsize=(35,35))
fig.set_facecolor('white')

for i in range(35, 36):
    train1 = train_set[train_set.Store == i] 
    plt.figure(figsize=(18, 18))
    sns.heatmap(data=train1.corr(), annot=True)
plt.show()

###### 데이터 전처리 ######

# 결측치 처리 

# Promotion NaN값 변환
train_set['Promotion1'] = train_set['Promotion1'].fillna(train_set['Promotion1'].mode()[0])
train_set['Promotion2'] = train_set['Promotion2'].fillna(train_set['Promotion2'].mode()[0])
train_set['Promotion3'] = train_set['Promotion3'].fillna(train_set['Promotion3'].mode()[0])
train_set['Promotion4'] = train_set['Promotion4'].fillna(train_set['Promotion4'].mode()[0])
train_set['Promotion5'] = train_set['Promotion5'].fillna(train_set['Promotion5'].mode()[0])
print(train_set.head(5))

test_set['Promotion1'] = test_set['Promotion1'].fillna(test_set['Promotion1'].mode()[0])
test_set['Promotion2'] = test_set['Promotion2'].fillna(test_set['Promotion2'].mode()[0])
test_set['Promotion3'] = test_set['Promotion3'].fillna(test_set['Promotion3'].mode()[0])
test_set['Promotion4'] = test_set['Promotion4'].fillna(test_set['Promotion4'].mode()[0])
test_set['Promotion5'] = test_set['Promotion5'].fillna(test_set['Promotion5'].mode()[0])
print(test_set.head(5))


# Date 날짜 변수 분리 
train_set['Date'] = pd.to_datetime(train_set['Date'])
train_set['year'] = train_set['Date'].dt.year
train_set['month'] = train_set['Date'].dt.month
train_set['week'] = train_set['Date'].dt.isocalendar().week
train_set['quarter'] = train_set['Date'].dt.quarter
train_set['day_of_week'] = train_set['Date'].dt.day_name()
print(train_set.head(5))

# Date 날짜 변수 분리 
test_set['Date'] = pd.to_datetime(test_set['Date'])
test_set['year'] = test_set['Date'].dt.year
test_set['month'] = test_set['Date'].dt.month
test_set['week'] = test_set['Date'].dt.isocalendar().week
test_set['quarter'] = test_set['Date'].dt.quarter
test_set['day_of_week'] = test_set['Date'].dt.day_name()
print(test_set.head(5))


# Date의 object 데이터를 int64 데이터로 변환 및 결측치 제거
cat_col = train_set.dtypes[train_set.dtypes == 'object'].index   
for col in cat_col:
    train_set[col] = train_set[col].fillna('None')
    train_set[col] = LabelEncoder().fit_transform(train_set[col].values)
print(train_set.head(5))       
  
cat_col = test_set.dtypes[test_set.dtypes == 'object'].index   
for col in cat_col:
    test_set[col] = test_set[col].fillna('None')
    test_set[col] = LabelEncoder().fit_transform(test_set[col].values)
print(test_set.head(5))     
  
train_set['IsHoliday'] = train_set['IsHoliday'].fillna(train_set.IsHoliday.dropna().mode()[0])
train_set['IsHoliday'] = train_set['IsHoliday'].apply(np.round).astype('float64')
test_set['IsHoliday'] = test_set['IsHoliday'].fillna(test_set.IsHoliday.dropna().mode()[0])
test_set['IsHoliday'] = test_set['IsHoliday'].apply(np.round).astype('float64')

print(train_set.shape, test_set.shape)      # (6255, 18) (180, 17)                                                 
print(train_set.head(), test_set.head())

# 정규분포와 가까운 모양으로 변환
num_col = list(train_set.dtypes[(train_set.dtypes == 'int64') 
                                 | (train_set.dtypes == 'float64')].index)
num_col = list(test_set.dtypes[(test_set.dtypes == 'int64') 
                                | (test_set.dtypes == 'float64')].index)

train_set[num_col].apply(lambda x : skew(x)).sort_values(ascending=False)
skewed_feat = train_set[num_col].apply(lambda x : skew(x)).sort_values(ascending=False)
skewed_feat = skewed_feat[abs(skewed_feat) > 0.75]
print(len(skewed_feat))

skewed_feat.index
skewed_feat = skewed_feat[abs(skewed_feat) > 0.75]
print(len(skewed_feat))

from scipy.special import boxcox1p
skewed_col = skewed_feat.index
for col in skewed_col:
    train_set[col] = boxcox1p(train_set[col], 0.15)   # 0.15는 lambda 값으로 변형정도를 결정한 값임

# Store와 Weekly_Sales와의 관계   
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9,6))
train_set.plot.line(x="Store", y="Weekly_Sales", color='b', title ="Store", ax = axes, rot=0)
plt.show()
   
# id, Date 제거        
train_set = train_set.drop(columns=['id', 'Date'])
test_set = test_set.drop(columns=['id', 'Date'])
print(train_set.head(), test_set.head)

# 데이터 확인
print(train_set.isnull().sum(), test_set.isnull().sum())
train_set['Promotion2'] = train_set['Promotion2'].fillna(train_set['Promotion2'].median())
train_set['Promotion3'] = train_set['Promotion3'].fillna(train_set['Promotion3'].median())
print(train_set.isnull().sum())
 
# drop_features
train_set = train_set.drop(['year', 'month'], axis=1)
test_set = test_set.drop(['year', 'month'], axis=1)

print("train_set.columns :", train_set.columns)
x = train_set.drop(['Weekly_Sales'], axis=1)
print(x)
print(x.columns)
# print(x.shape)  

y = train_set['Weekly_Sales']
print(y)
print(y.shape)

# train 데이터와 test 데이터의 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state =777)

# train 데이터와 test 데이터의 분리 결과 확인
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (6075, 13) (180, 13) (6075,) (180,)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='linear', input_dim=13))
model.add(Dropout(0.2))     
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))     
model.add(Dense(128, activation='relu'))
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
hist = model.fit(x_train, y_train, epochs=500, batch_size=64,
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

# Brutal approach to deal with predictions close to outer range 
q1 = submission['Weekly_Sales'].quantile(0.0045)
q2 = submission['Weekly_Sales'].quantile(0.99)

submission['Weekly_Sales'] = submission['Weekly_Sales'].apply(lambda x: x if x > q1 else x*0.77)
submission['Weekly_Sales'] = submission['Weekly_Sales'].apply(lambda x: x if x < q2 else x*1.1)

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
# loss :  [389015.4375, 247510417408.0]
# RMSE :  497504.1936765552
# R2 :  0.1773722212229869
#===================================================================================#
