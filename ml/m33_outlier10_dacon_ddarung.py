import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')


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
print(x.shape)  # (1328, 7)
y = train_set['count']

# IterativeImputer() 결측치 처리
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(random_state=72)
imputer.fit(x)
x = imputer.transform(x)

from sklearn.model_selection import train_test_split, KFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123
)

#2. 모델구성
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# model = RandomForestRegressor()
model = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
# print('model.score : ', result)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test,)
acc = r2_score(y_test, y_predict)
print('r2_score : ', acc)

#==================================== 결과 ==================================#
# 기존 r2 : 0.7450350233108447
# 결측치 및 이상치 처리 후 r2 : 0.7967343945309509
#============================================================================#