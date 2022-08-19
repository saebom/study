import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel


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
# 기존 :  0.7967343945309509
# 보팅 결과 :  0.7577
# CatBoostRegressor 정확도 : 0.7654
# XGBRegressor 정확도 : 0.7483
# LGBMRegressor 정확도 : 0.7115
#==============================================================================#