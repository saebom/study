import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel


#1. 데이터
path = './_data/house/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'house_submission.csv')
# print('train.shape, test.shape, submit.shape', 
#       train_set.shape, test_set.shape, submission.shape)    # (1460, 81) (1459, 80) (1459, 2)

###### 데이터 전처리 ######
print(train_set.describe(include=['number']).loc[['min','max','mean']].T.sort_values('max'))
print('after drop Id ::', train_set.shape, test_set.shape)  # (1460, 80) (1459, 79)

# 'SalePrice'와  'GrLivArea'의 관계
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(x = train_set['MSSubClass'], y = train_set['SalePrice'])
plt.ylabel('MSSubClass', fontsize = 13)
plt.ylabel('SalePrice', fontsize = 13)
plt.show()

# 이상치 제거 'SalePrice'와  'GrLivArea'
train_set = train_set.drop(train_set[(train_set['GrLivArea']>4000) 
                                     & (train_set['SalePrice']<300000)].index) 
train_set = train_set.drop(train_set[(train_set['TotalBsmtSF']>4000)].index) 

def outliers(df, col):
    out = []
    m = np.mean(df[col])
    sd = np.std(df[col])
    
    for i in df[col]: 
        z = (i-m)/sd
        if np.abs(z) > 3: 
            out.append(i)
            
    print("Outliers:", out)
    print("min",np.median(out))
    return np.median(out)

# all_data_set 데이터
label = train_set['SalePrice']
all_data_set = pd.concat((train_set, test_set)).reset_index(drop=True)
all_data_set = all_data_set.drop(['SalePrice'], axis=1)
print(all_data_set.shape)   # (2917, 79)
print(all_data_set)

# 결측값 조회
all_data_set_na = (all_data_set.isnull().sum() / len(all_data_set) * 100 ).sort_values(ascending=False)[ :25]
# print(all_data_set_na)

# ###(1) PoolQC 와 MiscFeature
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
cat_col = list(all_data_set.dtypes[all_data_set.dtypes == 'object'].index)  # 문자열로 된 feature 추출]
for col in cat_col:
    all_data_set[col] = LabelEncoder().fit_transform(all_data_set[col].values)

# 집의 건축물 종류를 구분하기 위해 숫자를 문자로 변경
all_data_set['MSSubClass'] = all_data_set['MSSubClass'].astype('category')

# 월을 계절로 변환하기 위해 숫자를 문자로 변경 (범주화하기)
all_data_set['MoSold'] = all_data_set['MoSold'].astype('category')

print(all_data_set['MoSold'].value_counts())    # category 생성되었는지 확인
print(all_data_set.dtypes)

# 집의 전체 넓이 확인
all_data_set['TotalSF'] = all_data_set['TotalBsmtSF'] + all_data_set['1stFlrSF'] + all_data_set['2ndFlrSF']

# 정규분포와 가까운 모양으로 변환
num_col = list(all_data_set.dtypes[(all_data_set.dtypes == 'int64') | (all_data_set.dtypes == 'float64')].index)
print(num_col)
print(all_data_set[num_col].head(3))    # 3줄까지 출력하여 확인

from scipy.stats import skew
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
    all_data_set[col] = boxcox1p(all_data_set[col], 0.5)   # 0.5는 lambda 값으로 변형정도를 결정한 값임
      
##### all_data_set 데이터 확인하기  
print(all_data_set.isnull().sum())
all_data_set['TotalSF'] = all_data_set['TotalSF'].fillna(all_data_set['TotalSF'].median())
print(all_data_set.shape)   # (2917, 80)


# 문자로 변경
all_data_set['MSSubClass'] = all_data_set['MSSubClass'].apply(str)
all_data_set['OverallCond'] = all_data_set['OverallCond'].astype(str)
all_data_set['YrSold'] = all_data_set['YrSold'].astype(str)
all_data_set['MoSold'] = all_data_set['MoSold'].astype(str)

# 라벨인코딩
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data_set[c].values)) 
    all_data_set[c] = lbl.transform(list(all_data_set[c].values))

# shape        
print('Shape all_data: {}'.format(all_data_set.shape))

# Adding total sqfootage feature 
all_data_set['TotalSF'] = all_data_set['TotalBsmtSF'] + all_data_set['1stFlrSF'] + all_data_set['2ndFlrSF']
numeric_feats = all_data_set.dtypes[all_data_set.dtypes != "object"].index

all_data = pd.get_dummies(all_data_set)
print(all_data_set.shape)

# all_data_set을 train_set과 test_set으로 분할
train_set = all_data_set[:len(train_set)]
test_set = all_data_set[len(train_set):]
print(train_set.shape, test_set.shape)  # (1458, 80) (1458, 80)

x = train_set
y = label   # train_set['SalePrice']

# IterativeImputer() 결측치 처리
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# imputer = SimpleImputer(strategy='most_frequent')   
imputer = IterativeImputer(random_state=72)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=72, train_size=0.8
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

xg = XGBRegressor(learning_rate=0.1, max_depth=3, random_state=1004, 
                   tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
lg = LGBMRegressor(learning_rate=0.3, max_depth=10, random_state=1004)
cat = CatBoostRegressor(verbose=0)

model = VotingRegressor(
    estimators=[('XG', xg), ('LG', lg), ('CAT', cat)],
    # voting='soft',   # hard
    # voting='hard',
    n_jobs=-1
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
score = r2_score(y_test, y_predict)
print('보팅 결과 : ', round(score, 4))

classifiers = [cat, xg, lg]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test, y_predict)
    class_name = model2.__class__.__name__
    print('{0} 정확도 : {1:.4f}'.format(class_name, score2), )

#=================================== 결과 =====================================#
# 기존 : 0.9054801606268293
# # 보팅 결과 :  0.917
# CatBoostRegressor 정확도 : 0.9202
# XGBRegressor 정확도 : 0.9051
# LGBMRegressor 정확도 : 0.8927
#==============================================================================#