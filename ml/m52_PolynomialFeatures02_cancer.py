import warnings
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_boston, load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline


#1. 데이터
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) #(569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=1234, train_size=0.8, shuffle=True, stratify=y
)

kfold = KFold(n_splits=5, shuffle=True, random_state=123)

#2. 모델
model = make_pipeline(StandardScaler(), DecisionTreeClassifier())

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
print(xp.shape)     # (506, 105)

x_train, x_test, y_train, y_test = train_test_split(
    xp, y, random_state=1234, train_size=0.8, shuffle=True, stratify=y
)

#2. 모델
model = make_pipeline(StandardScaler(), DecisionTreeClassifier())

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('폴리 sore : ', model.score(x_test, y_test))    # 0.8745129304823926

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("폴리 CV : ", scores)
print("폴리 CV 엔빵 : ", np.mean(scores))

#=================================== 결과 =====================================#
# 기냥 sore :  0.9736842105263158
# 기냥 CV :  [0.9010989  0.93406593 0.87912088 0.92307692 0.94505495]
# 기냥 CV 엔빵 :  0.9164835164835166
# 폴리 sore :  0.9736842105263158
# 폴리 CV :  [0.87912088 0.92307692 0.94505495 0.93406593 0.94505495]
# 폴리 CV 엔빵 :  0.9252747252747253
#==============================================================================#