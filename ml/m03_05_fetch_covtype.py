import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, accuracy_score


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, random_state=72
# )
n_splits = 9
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


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



#===================================== 결  과 ==========================================#
# LinearSVC() 결과 acc :  
# LogisticRegression() 결과 acc :  
# KNeighborsClassifier() 결과 acc :  
# DecisionTreeClassifier() 결과 acc :  
# RandomForestClassifier() 결과 acc :  0.9531909766844134
#=======================================================================================#

