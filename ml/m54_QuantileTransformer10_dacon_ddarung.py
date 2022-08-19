
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time


#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv',   # 예측에서 사용!!
                       index_col=0)

#### 결측치 처리 ####
test_set = test_set.fillna(method='ffill')
train_set = train_set.dropna()  # nan 값 삭제
print(train_set.shape, test_set.shape)  # (1328, 10) (715, 9)
print(train_set.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

# 'hour_bef_temperature'와  'hour_bef_pm10'의 관계
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(x = train_set['hour_bef_windspeed'], y = train_set['hour_bef_pm10'])
plt.ylabel('hour_bef_pm10', fontsize = 13)
plt.ylabel('hour_bef_windspeed', fontsize = 13)
plt.show()

# outliers 처리
def outliers(df, col):
    out = []
    m = np.mean(df[col])
    sd = np.std(df[col])
    
    for i in df[col]: 
        z = (i-m)/sd
        if np.abs(z) > 3: 
            out.append(i)
            
    print("Outliers:",out)
    print("min",np.median(out))
    return np.median(out)
    
col = "hour_bef_precipitation"
medOutlier = outliers(train_set, col)
train_set[train_set[col] >= medOutlier]
print(train_set[train_set[col] >= medOutlier])

col = "hour_bef_windspeed"
medOutlier = outliers(train_set,col)
train_set[train_set[col] >= medOutlier]

col = "hour_bef_ozone"
medOutlier = outliers(train_set,col)
train_set[train_set[col] >= medOutlier]


# x, y 데이터
x = train_set.drop(['count'], axis=1)
print(x.shape)  # (1328, 9)
y = train_set['count']

# IterativeImputer() 결측치 처리
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(random_state=72)
imputer.fit(x)
x = imputer.transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=72, train_size=0.8
)

# 스케일링
sts = StandardScaler() 
mms = MinMaxScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()
qtf = QuantileTransformer() 
ptf1 = PowerTransformer(method='yeo-johnson')
ptf2 = PowerTransformer(method='box-cox')

scalers = [sts, mms, mas, rbs, qtf, ptf1, ptf2]
for scaler in scalers:
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # model = LinearRegression()
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    result = r2_score(y_test, y_predict)
    scale_name = scaler.__class__.__name__
    print('{0} 결과 : {1:.4f}'.format(scale_name, result), )
    


#2. 모델
model = LinearRegression()
# model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
result = r2_score(y_test, y_predict)
print("기냥 결과 : ", round(result, 4)) 


#=================================== 결과 =====================================#
# StandardScaler 결과 : 0.7451
# MinMaxScaler 결과 : 0.7510
# MaxAbsScaler 결과 : 0.7416
# RobustScaler 결과 : 0.7503
# QuantileTransformer 결과 : 0.7472
# PowerTransformer 결과 : 0.7443
# ValueError: The Box-Cox transformation can only be applied to strictly positive data
#==============================================================================#