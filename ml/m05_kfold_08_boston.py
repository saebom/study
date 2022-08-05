import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVR, SVR


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, shuffle=True, random_state=66
# )

n_splits=9
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1004)


#2. 모델 구성
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()


#3.4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)
print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))


#==================================== KFold 적용 결과 ========================================================#
# ACC :  [0.78217898 0.78733826 0.90127758 0.8892864  0.90575042 0.9517406
#  0.75528143 0.85443751 0.87797112]
#  cross_val_score :  0.8561
#=============================================================================================================#
