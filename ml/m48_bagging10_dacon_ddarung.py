import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
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
    x, y, shuffle=True, random_state=123, train_size=0.8
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
from sklearn.svm import SVR
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor

# model = BaggingRegressor(SVR(), 
# model = BaggingRegressor(LinearRegression(), 
# model = BaggingRegressor(DecisionTreeRegressor(), 
model = BaggingRegressor(XGBRegressor(), 
                          n_estimators=100,
                          n_jobs=-1,
                          random_state=123
                          )

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()


#4. 평가, 예측
result = model.score(x_test, y_test)    
print('Bagging_DecisionTreeRegressor 결과 : ', result)
print('걸린 시간 : ', end - start)


#=================================== 결과 =====================================#
# 기존  r2 :  0.7967343945309509
# Bagging_XGBRegressor 결과 :  0.8093961974319118
# 걸린 시간 :  24.947882890701294
# Bagging_DecisionTreeRegressor 결과 :  0.7948974052349778
# 걸린 시간 :  9.95777678489685
# Bagging_DecisionTreeRegressor 결과 :  0.5851782948696027
# 걸린 시간 :  9.507796287536621 
# Bagging_SVR 결과 :  0.42374374368740597
# 걸린 시간 :  11.047969102859497
#==============================================================================#