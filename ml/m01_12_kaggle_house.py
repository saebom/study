import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from scipy.stats import norm, skew

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
train_set = train_set.drop(train_set[(train_set['TotalBsmtSF']>4000)].index) 

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

#=============================== 값변경 ===================================#
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data_set["LotFrontage"] = all_data_set.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data_set[col] = all_data_set[col].fillna('None')
    
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data_set[col] = all_data_set[col].fillna(0)
    
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data_set[col] = all_data_set[col].fillna(0)    
    
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data_set[col] = all_data_set[col].fillna('None')
    
all_data_set["MasVnrType"] = all_data_set["MasVnrType"].fillna("None")
all_data_set["MasVnrArea"] = all_data_set["MasVnrArea"].fillna(0)  
    
all_data_set['MSZoning'] = all_data_set['MSZoning'].fillna(all_data_set['MSZoning'].mode()[0])

all_data_set = all_data_set.drop(['Utilities'], axis=1)
all_data_set["Functional"] = all_data_set["Functional"].fillna("Typ")
all_data_set['Electrical'] = all_data_set['Electrical'].fillna(all_data_set['Electrical'].mode()[0])  
all_data_set['KitchenQual'] = all_data_set['KitchenQual'].fillna(all_data_set['KitchenQual'].mode()[0])
all_data_set['Exterior1st'] = all_data_set['Exterior1st'].fillna(all_data_set['Exterior1st'].mode()[0])
all_data_set['Exterior2nd'] = all_data_set['Exterior2nd'].fillna(all_data_set['Exterior2nd'].mode()[0])
all_data_set['SaleType'] = all_data_set['SaleType'].fillna(all_data_set['SaleType'].mode()[0])
# all_data_set['MSSubClass'] = all_data_set['MSSubClass'].fillna("None")

#Check remaining missing values if any 
all_data_na = (all_data_set.isnull().sum() / len(all_data_set)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()

#MSSubClass=The building class
all_data_set['MSSubClass'] = all_data_set['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data_set['OverallCond'] = all_data_set['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data_set['YrSold'] = all_data_set['YrSold'].astype(str)
all_data_set['MoSold'] = all_data_set['MoSold'].astype(str)

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

# Adding total sqfootage feature 
all_data_set['TotalSF'] = all_data_set['TotalBsmtSF'] + all_data_set['1stFlrSF'] + all_data_set['2ndFlrSF']

numeric_feats = all_data_set.dtypes[all_data_set.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data_set[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data_set[feat] = boxcox1p(all_data_set[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])
all_data = pd.get_dummies(all_data_set)
print(all_data_set.shape)

######## Model libraries import ########
# all_data_set을 train_set과 test_set으로 분할
train_set = all_data_set[:len(train_set)]
test_set = all_data_set[len(train_set):]
print(train_set.shape, test_set.shape)  # (1458, 80) (1458, 80)

x = train_set
y = label   # train_set['SalePrice']

print(x.shape, y.shape)
# train 데이터와 test 데이터의 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state =7)
# train 데이터와 test 데이터의 분리 결과 확인
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (1166, 80) (292, 80) (1166,) (292,)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
# model = Sequential()
# model.add(Dense(100, activation='linear', input_dim=79))
# model.add(Dense(100, activation ='relu'))
# model.add(Dense(100, activation ='relu'))
# model.add(Dense(100, activation ='relu'))
# model.add(Dense(100, activation ='relu'))
# model.add(Dense(1, activation='linear'))
from sklearn.svm import LinearSVR
model = LinearSVR()

#3. 컴파일, 훈련
# model.compile(loss='mae', optimizer='adam', metrics=['mse'])

# earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='min',
#                               restore_best_weights=True, 
#                               verbose=1)
# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=4000, batch_size=100,
#                  validation_split=0.2,
#                  callbacks=[earlyStopping],
#                  verbose=1)
# end_time = time.time() - start_time
model.fit(x_train, y_train)

#4. 평가예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)  
results = model.score(x_test, y_test)
print('결과 r2 : ', results)
      
# #### mse를 rmse로 변환 ####
# y_predict = model.predict(x_test)
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))

# rmse = RMSE(y_test, y_predict)
# print("RMSE : ", rmse)


# # r2 값 구하기  
# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2)  

# #### summit ####
# y_summit = model.predict(test_set)
# print(y_summit)
# print(y_summit.shape)   # (1458, 1)

# submission['SalePrice'] = y_summit

# # Brutal approach to deal with predictions close to outer range 
# q1 = submission['SalePrice'].quantile(0.0045)
# q2 = submission['SalePrice'].quantile(0.99)

# submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
# submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

# submission.to_csv(path + 'submission3.csv', index=False)


# print("=================================================================")   
# print(hist)     # <tensorflow.python.keras.callbacks.History object at 0x000002664FF27AF0>
# print("=================================================================")
# print(hist.history)     # loss 와 val_loss의 key, value를 합쳐놓은 것
# print("=================================================================")
# print(hist.history['loss'])
# print("=================================================================")
# print(hist.history['val_loss'])


#4_1. 그래프로 비교
# font_path = 'C:\Windows\Fonts\malgun.ttf'
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# # plt.title('loss & val_loss')    
# plt.title('로스값과 검증로스값')    
# plt.ylabel('loss')
# plt.xlabel('epochs')
# # plt.legend(loc='upper right')   # 우측상단에 라벨표시
# plt.legend()   # 자동으로 빈 공간에 라벨표시
# plt.show()

#================================ SVM 적용 결과 ===================================#
# 결과 r2 :  -2.3988554170670264
# =================================================================================
# loss : 18941.0
# mse : 921014208.0
# r2 스코어: 0.8968475516167238
#==================================================================================#

