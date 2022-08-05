import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, random_state=72
# )

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=0)

#2. 모델 구성
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression, LinearRegression     #LogisticRegression은 분류모델, LinearRegression 회귀모델
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# model = LinearSVR()
# model = LinearRegression()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()


#3.4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)
print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))



#=============================== StraitifiedKFold 적용 결과 ==================================================#
# ACC :  [0.80026051 0.80691603 0.81079821 0.81941211 0.8126803 ] 
#  cross_val_score :  0.81
#==================================== KFold 적용 결과 ========================================================#
# ACC :  [0.85044998 0.86817867 0.86278344 0.8569252  0.85014049]
#  cross_val_score :  0.8577
#============================================================================================================#

