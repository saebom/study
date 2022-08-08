import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=1234    
)


#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score : ', result)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test,)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)

print('==============================')
print(model, ': ', model.feature_importances_)


#============================================ 결과 ========================================================#
# model.score :  1.0
# accuracy_score :  1.0
# ==============================
# DecisionTreeClassifier() :  [0.         0.01669101 0.07659085 0.90671814]
# RandomForestClassifier() :  [0.09299566 0.02574918 0.43044751 0.45080764]
# GradientBoostingClassifier() :  [0.00554478 0.01440799 0.23833665 0.74171058]
# XGBClassifier :  [0.00912187 0.0219429  0.678874   0.29006115]
