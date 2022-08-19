import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.pipeline import make_pipeline


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

kfold = KFold(n_splits=5, shuffle=True, random_state=123)

#2. 모델
model = make_pipeline(StandardScaler(), LGBMRegressor())

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('기냥 sore : ', model.score(x_test, y_test))    # 0.7682699602324874

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print("기냥 CV : ", scores)
print("기냥 CV 엔빵 : ", np.mean(scores))

############################### PolynomialFeatures ##############################

pf = PolynomialFeatures(degree=2, 
                        include_bias=False
                        )
xp = pf.fit_transform(x)
print(xp.shape)     # (1328, 54)

x_train, x_test, y_train, y_test = train_test_split(
    xp, y, shuffle=True, random_state=72, train_size=0.8
)

#2. 모델
model = make_pipeline(StandardScaler(), LGBMRegressor())

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('폴리 sore : ', model.score(x_test, y_test))    # 0.8745129304823926

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print("폴리 CV : ", scores)
print("폴리 CV 엔빵 : ", np.mean(scores))

#=================================== 결과 =====================================#
# 기냥 sore :  0.7512892998425627
# 기냥 CV :  [0.77501896 0.73517934 0.75633081 0.80129517 0.74948786]
# 기냥 CV 엔빵 :  0.7634624268011925
# 폴리 sore :  0.7820730886990306
# 폴리 CV :  [0.76602799 0.77267878 0.77669896 0.81295086 0.77199341]
# 폴리 CV 엔빵 :  0.7800699989272475
#==============================================================================#