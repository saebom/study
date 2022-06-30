# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

# Kaggle house prices 문제풀이

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from scipy.stats import skew


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
print(train_set['YrSold'])
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
y = train_set['SalePrice']
all_data_set = pd.concat((train_set, test_set)).reset_index(drop=True)
all_data_set = all_data_set.drop(['SalePrice'], axis=1)
print(all_data_set.shape)   # (2917, 79)
print(all_data_set)


# 결측값 조회
all_data_set_na = (all_data_set.isnull().sum() / len(all_data_set) * 100 ).sort_values(ascending=False)[ :25]
# print(all_data_set_na)

## 3) missing value 넣기
# PoolQC은 주변에 거지가 없다는 것을 의미하는 변수인데 이 feature는 거의 99퍼 null임. 사실 집 근처에 거지가 있는 경우는 거의 없으므로 
# fillna함수를 이용해 누락된 값에 모두 None을 넣음
all_data_set["PoolQC"] = all_data_set["PoolQC"].fillna("None")
# 밑에 쥐가 있는지, 골목이 있는지, 담장이 있는지, 난로가 없는지도 모두 동일한 맥락에서 None넣음.
all_data_set["MiscFeature"] = all_data_set["MiscFeature"].fillna("None")
all_data_set["Alley"] = all_data_set["Alley"].fillna("None")
all_data_set["Fence"] = all_data_set["Fence"].fillna("None")
all_data_set["FireplaceQu"] = all_data_set["FireplaceQu"].fillna("None")

# 인접한 길의 직선거리는 인접 주택과 비슷하므로 이웃이 지는 값의 평균으로
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data_set["LotFrontage"] = all_data_set.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# 차고 관련 feature들은 매우작으므로 문자열로 나타내어지는 것들은 None, 숫자인 것들은 0으로. 
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data_set[col] = all_data_set[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data_set[col] = all_data_set[col].fillna(0)

# 지하관련 feature들도 차고 관련된 것드로가 마찬가지
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data_set[col] = all_data_set[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data_set[col] = all_data_set[col].fillna('None')

# 벽돌관련 feature도 마찬가지로
all_data_set["MasVnrType"] = all_data_set["MasVnrType"].fillna("None")
all_data_set["MasVnrArea"] = all_data_set["MasVnrArea"].fillna(0)

# 일반적인 구역분류(?) 라는 feature인데 이 feature은 대부분 RL이라서 첫 번째 MSZoing인 RL을 fillna함.
all_data_set['MSZoning'] = all_data_set['MSZoning'].fillna(all_data_set['MSZoning'].mode()[0])

# utilities는 두 개 말고는 allpub으로 통일되어있어서 유의미한 feature가 되지 못함. 싹 다 지우기
all_data_set = all_data_set.drop(['Utilities'], axis=1)

# TYP가 대부분. 그래서 NULL 이걸로 채우기.
all_data_set["Functional"] = all_data_set["Functional"].fillna("Typ")

# 대부분이 SBrkr이고 NA가 1개뿐이라 채우기
all_data_set['Electrical'] = all_data_set['Electrical'].fillna(all_data_set['Electrical'].mode()[0])

# 이것도 하나 비어있음.
all_data_set['KitchenQual'] = all_data_set['KitchenQual'].fillna(all_data_set['KitchenQual'].mode()[0])

# 이것도 하나 비어있음.
all_data_set['Exterior1st'] = all_data_set['Exterior1st'].fillna(all_data_set['Exterior1st'].mode()[0])
all_data_set['Exterior2nd'] = all_data_set['Exterior2nd'].fillna(all_data_set['Exterior2nd'].mode()[0])

# WD가 대부분. 그래서 NULL 이걸로 채우기.
all_data_set['SaleType'] = all_data_set['SaleType'].fillna(all_data_set['SaleType'].mode()[0])

# BuildingClass가 높이를 뜻하는지 등급을 뜻하는지 명확하지 않지만 null이면 보통 none으로 봐야한다고 설명되어있음. 
all_data_set['MSSubClass'] = all_data_set['MSSubClass'].fillna("None")

#Check remaining missing values if any 
all_data_na = (all_data_set.isnull().sum() / len(all_data_set)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()

# print(all_data_set)
print(all_data_set.info())  # ==> NaN값 해결됨

# (4) more feature enginnering
#  1) Transforming some numerical variables that are really categorical(범주화할 수 있는 변수들을 범주화하기)
#MSSubClass=The building class
all_data_set['MSSubClass'] = all_data_set['MSSubClass'].apply(str)

#Changing OverallCond into a categorical variable
all_data_set['OverallCond'] = all_data_set['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
all_data_set['YrSold'] = all_data_set['YrSold'].astype(str)
all_data_set['MoSold'] = all_data_set['MoSold'].astype(str)


# 모든 데이터의 문자를 숫자로 변경
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data_set[c].values)) 
    all_data_set[c] = lbl.transform(list(all_data_set[c].values))

# shape        
print('Shape all_data: {}'.format(all_data_set.shape))

# 집의 전체 넓이 확인
all_data_set['TotalSF'] = all_data_set['TotalBsmtSF'] + all_data_set['1stFlrSF'] + all_data_set['2ndFlrSF']

# 정규분포와 가까운 모양으로 변환
num_col = list(all_data_set.dtypes[(all_data_set.dtypes == 'int64') | (all_data_set.dtypes == 'float64')].index)
# print(num_col)
# print(all_data_set[num_col].head(3))    # 3줄까지 출력하여 확인

# 4) Skewed features
numeric_feats = all_data_set.dtypes[all_data_set.dtypes != "object"].index
all_data_set[num_col].apply(lambda x : skew(x)).sort_values(ascending=False)
skewed_feat = all_data_set[num_col].apply(lambda x : skew(x)).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feat})
skewed_feat = skewed_feat[abs(skewed_feat) > 0.75]
print(len(skewed_feat))

from scipy.special import boxcox1p
skewed_col = skewed_feat.index
for col in skewed_col:
    all_data_set[col] = boxcox1p(all_data_set[col], 0.15)   # 0.15는 lambda 값으로 변형정도를 결정한 값임
   
all_data_set = pd.get_dummies(all_data_set)

##### all_data_set 데이터 확인하기  
print(all_data_set.shape)  

# all_data_set을 train_set과 test_set으로 분할
train_set = all_data_set[:len(train_set)]
test_set = all_data_set[len(train_set):]
print(train_set.shape, test_set.shape)  # (1458, 80) (1458, 80)

######## Model libraries import ########
# all_data_set을 train_set과 test_set으로 분할
train_set = all_data_set[:len(train_set)]
test_set = all_data_set[len(train_set):]
print(train_set.shape, test_set.shape)  # (1458, 80) (1458, 80)

x = train_set  
# y = train_set['SalePrice']

print(x.shape, y.shape)
# train 데이터와 test 데이터의 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state =7)

# train 데이터와 test 데이터의 분리 결과 확인
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (1166, 80) (292, 80) (1166,) (292,)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=220))
model.add(Dense(100, activation ='elu'))
model.add(Dense(100, activation ='elu'))
model.add(Dense(100, activation ='elu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=5000, batch_size=128, verbose=0)

#4. 평가예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)  
    
#### mse를 rmse로 변환 ####
y_predict = model.predict(x_test)

def rmsle(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))   
    
rmsle = rmsle(y_test, y_predict)
print("RMSLE : ", rmsle)    
  
# r2 값 구하기  
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)  

#### summit ####
y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape)   # (1458, 1)

# Brutal approach to deal with predictions close to outer range 
q1 = y_summit.quantile(0.0045)  # 0.0045 분위에 위치한 값
q2 = y_summit.quantile(0.999)

y_summit = y_summit.apply(lambda x: x if x > q1 else x*0.77)
y_summit = y_summit.apply(lambda x: x if x < q2 else x*1.1)

submission['SalePrice'] = y_summit
submission.to_csv(path + 'submission1.csv', index=False)


# loss :  133640960.0
# RMSLE :  11560.314325624347
# R2 :  0.9850323875991999
# submission.csv
# Score: 0.14982
  
# loss :  45597864.0
# RMSLE :  6752.622107549778
# R2 :  0.9948930908113651  
# submission1.csv
# Score: 0.17


