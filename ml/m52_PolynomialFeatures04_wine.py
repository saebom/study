import warnings
import numpy as np
from sklearn.datasets import load_wine, load_boston, load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=2022, shuffle=True
)

kfold = KFold(n_splits=5, shuffle=True, random_state=123)

#2. 모델
model = make_pipeline(StandardScaler(), LGBMClassifier())

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('기냥 sore : ', model.score(x_test, y_test))    # 0.7682699602324874

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("기냥 CV : ", scores)
print("기냥 CV 엔빵 : ", np.mean(scores))

############################### PolynomialFeatures ##############################

pf = PolynomialFeatures(degree=2, 
                        include_bias=False
                        )
xp = pf.fit_transform(x)
print(xp.shape)     # (150, 5)

x_train, x_test, y_train, y_test = train_test_split(
    xp, y, shuffle=True, random_state=2022, train_size=0.8
)

#2. 모델
model = make_pipeline(StandardScaler(), LGBMClassifier())

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('폴리 sore : ', model.score(x_test, y_test))   
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("폴리 CV : ", scores)
print("폴리 CV 엔빵 : ", np.mean(scores))

#=================================== 결과 =====================================#
# 기냥 sore :  0.9722222222222222
# 기냥 CV :  [0.96551724 0.93103448 0.96428571 1.         1.        ]
# 기냥 CV 엔빵 :  0.9721674876847292
# 폴리 sore :  0.9722222222222222
# 폴리 CV :  [0.96551724 0.96551724 0.96428571 1.         1.        ]
# 폴리 CV 엔빵 :  0.9790640394088669
#==============================================================================#