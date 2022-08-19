import warnings
import numpy as np
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
datasets = fetch_california_housing()
x, y = datasets.data, datasets.target
print(x.shape, y.shape)     # (20640, 8) (20640,) 

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
print(xp.shape)     # (20640, 44)

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
# 기냥 sore :  0.8432447134059025
# 기냥 CV :  [0.83521026 0.83622163 0.82989538 0.84104352 0.8248937 ]
# 기냥 CV 엔빵 :  0.8334528978340746
# 폴리 sore :  0.8404106067642194
# 폴리 CV :  [0.83852754 0.84152487 0.8314538  0.8380561  0.82240264]
# 폴리 CV 엔빵 :  0.8343929886547438
#==============================================================================#