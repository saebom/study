# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

# Kaggle house prices 문제풀이

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from scipy.stats import norm, skew
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))  # limiting floats output to 3 decimal points

#1. 데이터
path = './_data/house/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_set)
# print(train_set.columns)                         
# print('train_set의 info :: ', train_set.info())
# print(train_set.describe())
# print(train_set.isnull().sum())

test_set = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_set) 
# print(test_set.columns) 
# print('test_set의 info :: ', test_set.info())
# print(test_set.describe())
# print(test_set.isnull().sum())  # LotFrontage 227, SaleType 1

submission = pd.read_csv(path + 'house_submission.csv')
print('train.shape, test.shape, submit.shape', 
      train_set.shape, test_set.shape, submission.shape)    # (1460, 81) (1459, 80) (1459, 2)


###### 데이터 전처리 ######
print('after drop Id ::', train_set.shape, test_set.shape)  # (1460, 80) (1459, 79)

# 'SalePrice'와  'GrLivArea'의 관계
# fig, ax = plt.subplots()
# ax.scatter(x = train_set['GrLivArea'], y = train_set['SalePrice'])
# plt.ylabel('SalePrice', fontsize = 13)
# plt.ylabel('GrLivArea', fontsize = 13)
# plt.show()

# 'SalePrice'와  'GrLivArea'의 이상치 제거
train_set = train_set.drop(train_set[(train_set['GrLivArea']>4000) 
                                     & (train_set['SalePrice']<300000)].index) 

# 'SalePrice'와  'GrLivArea'의 관계
# fig, ax = plt.subplots()
# ax.scatter(x = train_set['GrLivArea'], y = train_set['SalePrice'])
# plt.ylabel('SalePrice', fontsize = 13)
# plt.ylabel('GrLivArea', fontsize = 13)
# plt.show()

# sns.displot(train_set['SalePrice'], fit=norm)

# 함수에 사용하기 위한 전처리 및 plot
# (mu, sigma) = norm.fit(train_set['SalePrice'])
# print('\n mu = {:.2f} and sigma = {:.2f}\n'. format(mu, sigma)) # mu = 180932.92 and sigma = 79467.79

# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'. format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')


# SalePrice 제거
label = train_set['SalePrice']
all_data_set = pd.concat((train_set, test_set)).reset_index(drop=True)
all_data_set = all_data_set.drop(['SalePrice'], axis=1)
print(all_data_set.shape)   # (2917, 79)
print(all_data_set)


# 결측값 조회
all_data_set_na = (all_data_set.isnull().sum() / len(all_data_set) * 100 ).sort_values(ascending=False)[ :25]
# print(all_data_set_na)

# ###(1) PoolQC 와 MiscFeature
# print(all_data_set['PoolQC'])
# print(all_data_set['MiscFeature'])
all_data_set['PoolQC'] = all_data_set['PoolQC'].fillna('None')
all_data_set['MiscFeature'] = all_data_set['MiscFeature'].fillna('None')

###(2) LotFrontage 와 KitchenQual
all_data_set['LotFrontage'].isnull().sum()
all_data_set['LotFrontage'].value_counts()
all_data_set['LotFrontage'] = all_data_set['LotFrontage'].fillna(all_data_set['LotFrontage'].median())
all_data_set['KitchenQual'].value_counts()
for col in ['MSZoning', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']:
    all_data_set[col] = all_data_set[col].fillna(all_data_set[col].mode()[0])

# print(all_data_set)
print(all_data_set.info())  # ==> NaN값 해결됨

# 모든 데이터 즉 object 데이터, int64 및 float64 데이터의 결측치 제거
cat_col = all_data_set.dtypes[all_data_set.dtypes == 'object'].index   
for col in cat_col:
    all_data_set[col] = all_data_set[col].fillna('None')

col_count = all_data_set.dtypes[(all_data_set.dtypes == 'int64') | (all_data_set.dtypes == 'float64')].index
for col in col_count:
    all_data_set[col] = all_data_set[col].fillna(0)
    
(all_data_set.isnull().sum() / len(all_data_set) * 100).sort_values(ascending=False)[:5]    # 결측치 제거 확인
print(all_data_set.head(5))  

# 모든 데이터의 문자를 숫자로 변경
from sklearn.preprocessing import LabelEncoder
# label_1 = LabelEncoder()
cat_col = list(all_data_set.dtypes[all_data_set.dtypes == 'object'].index)  # 문자열로 된 feature 추출]
for col in cat_col:
    # all_data_set[col] = label_1.fit_transform(all_data_set[col].values)
    all_data_set[col] = LabelEncoder().fit_transform(all_data_set[col].values)
# print(all_data_set.head(2))     # ==> 2줄까지 출력하여 확인

all_data_set['MSSubClass'] = all_data_set['MSSubClass'].astype('category')

# 집의 건축물 종류를 구분하기 위해 숫자를 문자로 변경
all_data_set['MSSubClass'] = all_data_set['MSSubClass'].astype('category')

# 월을 계절로 변환하기 위해 숫자를 문자로 변경 (범주화하기)
all_data_set['MoSold'] = all_data_set['MoSold'].astype('category')

# print(all_data_set['MoSold'].value_counts())    # category 생성되었는지 확인
# print(all_data_set.dtypes)

# 집의 전체 넓이 확인
all_data_set['TotalSF'] = all_data_set['TotalBsmtSF'] + all_data_set['1stFlrSF'] + all_data_set['2ndFlrSF']

# 정규분포와 가까운 모양으로 변환
num_col = list(all_data_set.dtypes[(all_data_set.dtypes == 'int64') | (all_data_set.dtypes == 'float64')].index)
# print(num_col)
# print(all_data_set[num_col].head(3))    # 3줄까지 출력하여 확인

all_data_set[num_col].apply(lambda x : skew(x)).sort_values(ascending=False)
skewed_feat = all_data_set[num_col].apply(lambda x : skew(x)).sort_values(ascending=False)
skewed_feat = skewed_feat[abs(skewed_feat) > 0.75]
print(len(skewed_feat))

skewed_feat.index
skewed_feat = skewed_feat[abs(skewed_feat) > 0.75]
print(len(skewed_feat))

from scipy.special import boxcox1p
skewed_col = skewed_feat.index
for col in skewed_col:
    all_data_set[col] = boxcox1p(all_data_set[col], 0.15)   # 0.15는 lambda 값으로 변형정도를 결정한 값임
   
   
##### all_data_set 데이터 확인하기  
print(all_data_set.isnull().sum())
all_data_set['TotalSF'] = all_data_set['TotalSF'].fillna(all_data_set['TotalSF'].median())
print(all_data_set.shape)   # (2917, 80)


######## Model libraries import ########
# all_data_set을 train_set과 test_set으로 분할
train_set = all_data_set[:len(train_set)]
test_set = all_data_set[len(train_set):]
print(train_set.shape, test_set.shape)  # (1458, 80) (1458, 80)

x = train_set
y = label   # train_set['SalePrice']

print(x.shape, y.shape)
# train 데이터와 test 데이터의 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=17)   # random_state=113

# train 데이터와 test 데이터의 분리 결과 확인
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (1166, 80) (292, 80) (1166,) (292,)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=80))
model.add(Dense(100, activation ='selu'))
model.add(Dense(100, activation ='selu'))
model.add(Dense(100, activation ='selu'))
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

#### rmsle 변환 ####
from sklearn.metrics import make_scorer

def rmsle(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)

    log_test = np.log(y_test + 1)
    log_predict = np.log(y_predict + 1)

    # 1) 예측값에서 실측값을 빼 준 후 제곱해 줌
    difference = log_predict - log_test
    difference = np.square(difference)

    # 2) 평균값 구함
    mean_difference = difference.mean()

    # 3) 다시 루트 씌움
    score = np.sqrt(mean_difference)

    return score    
    
rmsle = rmsle(y_test, y_predict)
print("RMSLE : ", rmsle)    
  
# r2 값 구하기  
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)  
    
#### summit ####
y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape)   # (1458, 1)

submission['SalePrice'] = y_summit
submission.to_csv(path + 'submission.csv', index=False)

# loss :  53163376.0
# RMSE :  7291.322253906671
# RMSLE :  0.5497207170405027
# R2 :  0.9909261706168205

# loss :  57947928.0
# RMSE :  7612.356908250641
# RMSLE :  0.5825278990202705
# R2 :  0.9911157414332816