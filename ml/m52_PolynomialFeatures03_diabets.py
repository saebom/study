import warnings
import numpy as np
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

x = np.delete(x, [0, 1, 5], axis=1)
print(x.shape)  # (442, 8)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=72, train_size=0.8
)

kfold = KFold(n_splits=5, shuffle=True, random_state=123)

#2. 모델
model = make_pipeline(StandardScaler(), LinearRegression())

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
print(xp.shape)     # (442, 35)

x_train, x_test, y_train, y_test = train_test_split(
    xp, y, shuffle=True, random_state=72, train_size=0.8
)

#2. 모델
model = make_pipeline(StandardScaler(), LinearRegression())

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('폴리 sore : ', model.score(x_test, y_test))    

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print("폴리 CV : ", scores)
print("폴리 CV 엔빵 : ", np.mean(scores))

#=================================== 결과 =====================================#
# 기냥 sore :  0.6276838952338201
# 기냥 CV :  [0.45399804 0.46526046 0.3004489  0.42420403 0.40921318]
# 기냥 CV 엔빵 :  0.4106249191711691
# 폴리 sore :  0.5262718938752424
# 폴리 CV :  [0.41066319 0.41854945 0.06499453 0.38140192 0.2999774 ]
# 폴리 CV 엔빵 :  0.3151172973239728
#==============================================================================#