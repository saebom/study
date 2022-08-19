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
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.pipeline import make_pipeline


#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target
print(x.shape, y.shape)     # (506, 13) (506,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=2020, train_size=0.8
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
print(xp.shape)     # (506, 105)

x_train, x_test, y_train, y_test = train_test_split(
    xp, y, shuffle=True, random_state=1234, train_size=0.8
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
# 기냥 sore :  0.8389861191947823
# 기냥 CV :  [0.83565702 0.82921623 0.88966223 0.83963475 0.85872958]
# 기냥 CV 엔빵 :  0.8505799605299721
# 폴리 sore :  0.9042279662517133
# 폴리 CV :  [0.90910673 0.84649257 0.90413295 0.86767299 0.75299364]
# 폴리 CV 엔빵 :  0.8560797752829721
#==============================================================================#