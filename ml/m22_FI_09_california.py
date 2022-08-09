import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# drop_features
x = np.delete(x, [3, 4], axis=1)
print(x.shape)  # (20640, 6)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72  
)


#2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor()

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
# 기존 r2 : 0.57540700823129
# 컬럼 삭제 후 r2 : 0.6059611789394723
# [0.51793108 0.05146041 0.0544596  0.02856055 0.03237592 0.13713418
#  0.08720992 0.09086833]

# 2. RandomForestRegressor
# 기존 r2 : 0.8151749843480731
# 컬럼 삭제 후 r2 : 0.8164165443678535
# [0.51513792 0.05165428 0.04348626 0.02924218 0.03134076 0.14193391
#  0.09397882 0.09322587]

# 3. GradientBoostingRegressor
# 기존 r2 :  0.7925066805006563
# 컬럼 삭제 후 r2 : 0.7941498498012171
#  [0.59344533 0.03150394 0.02380145 0.00366859 0.00403267 0.12962739
#  0.10117418 0.11274646]

# 4. XGBRegressor
# 기존 r2 : 0.8333568876916249
# 컬럼 삭제 후 r2 : 0.842194435829795
# [0.46827975 0.06449839 0.04786681 0.02352507 0.02535573 0.15400264
#  0.10176818 0.1147034 ]
#=========================================================================================================================#
