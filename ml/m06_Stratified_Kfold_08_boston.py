import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVR, SVR


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=66
)

n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1004)


#2. 모델 구성
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Perceptron, LogisticRegression     #LogisticRegression은 분류모델
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model =LinearSVR()


#3.4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2, )

#=============================== StraitifiedKFold 적용 결과 ==================================================#
# ValueError: Supported target types are: ('binary', 'multiclass'). Got 'multilabel-indicator' instead.
# 회귀모델은 StratifiedKfold 적용이 안됨
#==================================== KFold 적용 결과 ========================================================#
# ACC :  [0.78217898 0.78733826 0.90127758 0.8892864  0.90575042 0.9517406
#  0.75528143 0.85443751 0.87797112]
#  cross_val_score :  0.8561
#=============================================================================================================#
