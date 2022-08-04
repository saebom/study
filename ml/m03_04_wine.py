import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.svm import LinearSVC


#1. 데이터
datasets = load_wine()
x, y = datasets.data, datasets.target

print(x.shape)  # (178, 13)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression     #LogisticRegression은 분류모델
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = LinearSVC()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 acc : ', result)


#===================================== 결  과 ==========================================#
# LinearSVC() 결과 acc : 0.9259259259259259
# LogisticRegression() 결과 acc :  0.9629629629629629
# KNeighborsClassifier() 결과 acc :  0.6851851851851852
# DecisionTreeClassifier() 결과 acc :  0.9444444444444444
# RandomForestClassifier() 결과 acc :  1.0
#=======================================================================================#


