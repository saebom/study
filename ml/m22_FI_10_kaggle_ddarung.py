from cgi import test
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


# drop_features
train_set = train_set.drop(['hour_bef_temperature', 'hour_bef_ozone'], axis=1)
print(train_set.shape)  # (1328, 8)
test_set = test_set.drop(['hour_bef_temperature', 'hour_bef_ozone'], axis=1)
print(test_set.shape)  # (715, 7)



x = train_set.drop(['count'], axis=1)
print(x.shape)  # (1328, 7)
y = train_set['count']

from sklearn.model_selection import train_test_split, KFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72
)

#2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
# model = XGBRegressor()

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
# print('model.score : ', result)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test,)
acc = r2_score(y_test, y_predict)
print('r2_score : ', acc)

print('==============================')
print(model, ': ', model.feature_importances_)



#결과비교

#======================================  [] 삭제  결과 =======================================#
# 1. DecisionTreeRegressor
# 기존 r2 : 0.43254523804461076
# 컬럼 삭제 후 r2 : 0.3752589131933791
# [0.61945098 0.17415238 0.02894124 0.03424656 0.03753698 0.03392298
#  0.01237986 0.03643762 0.02293139]

# 2. RandomForestRegressor
# 기존 r2 : 0.7450350233108447
# 컬럼 삭제 후 r2 : 0.6562025533546171
# [0.59291614 0.17700278 0.01860974 0.03058929 0.04067891 0.03846597
#  0.03932567 0.03767607 0.02473544]

# 3. GradientBoostingRegressor
# 기존 r2 :  0.7474763418166968
# 컬럼 삭제 후 r2 : 0.6808405667922739
# [0.6533222  0.21222425 0.02471186 0.01225513 0.01235645 0.02894762
#  0.02612282 0.02114831 0.00891135]

# 4. XGBRegressor
# 기존 r2 : 0.7066938560926904
# 컬럼 삭제 후 r2 : 0.6469350185457399
# [0.32377088 0.08324073 0.43909505 0.01729347 0.02577747 0.02560352
#  0.02950996 0.02856151 0.02714741]
#=========================================================================================================================#
