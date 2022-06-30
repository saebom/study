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
y_train = train_set['SalePrice']
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

#2. 모델구성
#######  https://www.kaggle.com/code/kwakjunewon/house-price-practice-2-staked-regression  ##############  
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import time


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_set.values)
    rmse= np.sqrt(-cross_val_score(model, train_set.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(X) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1) 
    
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))
score = rmsle_cv(averaged_models)
score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
# We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
       # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

    
# Stacking Averaged models Score(이제 staking한 모델로 점수내보기)
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


def rmsle(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))   


stacked_averaged_models.fit(train_set.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train_set.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(train_set.values))
print(rmsle(y_train, stacked_train_pred))


lasso.fit(train_set, y_train)
lasso_result_train_pred = lasso.predict(train_set)
lasso_result_pred = np.expm1(lasso.predict(test_set))
print(rmsle(y_train, lasso_result_train_pred))
# (2) XGboost로
model_xgb.fit(train_set, y_train)
xgb_train_pred = model_xgb.predict(train_set)
xgb_pred = np.expm1(model_xgb.predict(y_train))
print(rmsle(y_train, xgb_train_pred))
# (3) LightGBM로
model_lgb.fit(train_set, y_train)
lgb_train_pred = model_lgb.predict(train_set)
lgb_pred = np.expm1(model_lgb.predict(test_set.values))
print(rmsle(y_train, lgb_train_pred))
# (4) 최종적으로 저 위의 3개를 ensemble한 값
'''RMSE on the entire Train data when averaging'''

print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15))
      
      
print(rmsle(y_train,stacked_train_pred*0.40 +
               xgb_train_pred*0.30+lgb_train_pred*0.30))

# ensemble해서
ensemble = stacked_pred*0.40 + xgb_pred*0.3 + lgb_pred*0.3
# submission파일 만듬. 
submission = pd.DataFrame()
submission['SalePrice'] = ensemble
submission.to_csv(path + 'submission1.csv',index=False)

'''     
model = Sequential()
model.add(Dense(100, input_dim=79))
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


    
rmsle = rmsle(y_test, y_predict)
print("RMSLE : ", rmsle)    
  

def rmsle(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    


stacked_averaged_models.fit(train_set.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train_set.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test_set.values))
print(rmsle(y_train, stacked_train_pred))
 
 
model_xgb.fit(train_set, y_train)
xgb_train_pred = model_xgb.predict(train_set)
xgb_pred = np.expm1(model_xgb.predict(test_set))
print(rmsle(y_train, xgb_train_pred)) 

model_lgb.fit(train_set, y_train)
lgb_train_pred = model_lgb.predict(train_set)
lgb_pred = np.expm1(model_lgb.predict(test_set))
print(rmsle(y_train, lgb_train_pred))

# RMSE on the entire Train data when averaging

print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15

####################### 


# r2 값 구하기  
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)  

#### summit ####
y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape)   # (1458, 1)

submission['SalePrice'] = y_summit

# Brutal approach to deal with predictions close to outer range 
q1 = submission['SalePrice'].quantile(0.0045)
q2 = submission['SalePrice'].quantile(0.99)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv(path + 'submission.csv', index=False)

'''
# loss :  53163376.0
# RMSE :  7291.322253906671
# RMSLE :  0.5497207170405027
# R2 :  0.9909261706168205
  