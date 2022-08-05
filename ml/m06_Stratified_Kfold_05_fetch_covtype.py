import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import r2_score, accuracy_score


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, random_state=72
# )
n_splits = 9
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)


#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression     #LogisticRegression은 분류모델
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = LinearSVC()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()


#3.4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)
print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))



#=============================== StraitifiedKFold 적용 결과 ==================================================#
# # ACC :  [0.95715414 0.9576963  0.95737101 0.95610081 0.95545022 0.95794414
#  0.9587961  0.95902846 0.95738584]
#  cross_val_score :  0.9574
#==================================== KFold 적용 결과 ========================================================#
# ACC :  [0.95726257 0.95809904 0.95744846 0.95579101 0.95734003 0.95830042
#  0.9567514  0.95766532 0.95927567]
#  cross_val_score :  0.9575
#=============================================================================================================#

