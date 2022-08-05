import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils import all_estimators


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, random_state=66
# )

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression     #LogisticRegression은 분류모델
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = SVC()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()


#3.4. 컴파일, 훈련, 평가, 예측
# model.fit(x_train, y_train) 
scores = cross_val_score(model, x, y, cv=kfold)
# scores = cross_val_score(model, x, y, cv=5)
print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))


#===================================== 결  과 ==========================================#
# ACC :  [0.96666667 0.96666667 1.         0.93333333 0.96666667] 
#  cross_val_score :  0.9667
#=======================================================================================#


