import warnings
import numpy as np
from sklearn.datasets import load_wine, load_digits, load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
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
datasets = load_digits()
x = datasets.data
y = datasets.target     # (1797, 64) (1797,)
print(x.shape, y.shape)     

# 라벨인코딩
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72, shuffle=True
)

kfold = KFold(n_splits=5, shuffle=True, random_state=123)

#2. 모델
model = make_pipeline(StandardScaler(), LGBMClassifier())

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('기냥 sore : ', model.score(x_test, y_test))   

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("기냥 CV : ", scores)
print("기냥 CV 엔빵 : ", np.mean(scores))

############################### PolynomialFeatures ##############################

pf = PolynomialFeatures(degree=2, 
                        include_bias=False
                        )
xp = pf.fit_transform(x)
print(xp.shape)     # (1797, 2144)    

x_train, x_test, y_train, y_test = train_test_split(
    xp, y, shuffle=True, random_state=72, train_size=0.8
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
# 기냥 sore :  0.9833333333333333
# 기냥 CV :  [0.96875    0.95833333 0.97560976 0.97212544 0.97212544]
# 기냥 CV 엔빵 :  0.9693887921022067
# 폴리 sore :  0.9833333333333333
# 폴리 CV :  [0.96875    0.96180556 0.96167247 0.96167247 0.95121951]
# 폴리 CV 엔빵 :  0.9610240030971738
#==============================================================================#