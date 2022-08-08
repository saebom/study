# 실습
# 피처임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여
# 데이터 셋 재구성한 후 
# 각 모델별로 돌려서 결과 도출 !

# 기존 모델결과와 비교

import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
print(datasets.DESCR)
print(x.shape)  # (150, 4)
# - sepal length in cm
# - sepal width in cm
# - petal length in cm
# - petal width in cm

# drop_features
x = np.delete(x, 0, axis=1)
print(x.shape)  # (150, 3)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72  
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



#결과비교

#============================================== 1열(sepal length) 삭제  결과 =============================================#
# 1. DecisionTree
# 기존 acc : 1.0
# 컬럼 삭제 후 acc : 1.0
# DecisionTreeClassifier() :  [0.01669101 0.07814299 0.905166  ]

# 2. RandomForestClassifier
# 기존 acc : 1.0
# 컬럼 삭제 후 acc : 1.0
# RandomForestClassifier() :  [0.11731571 0.42726147 0.45542282]

# 3. GradientBoostingClassifier
# 기존 acc : 1.0
# 컬럼 삭제 후 acc : 1.0
# GradientBoostingClassifier() :  [0.02026579 0.30569661 0.6740376 ]

# 4. XGBClassifier
# 기존 acc : 1.0
# 컬럼 삭제 후 acc : 1.0
# XGBClassifier() :  [0.02544769 0.66221166 0.31234065]
#=========================================================================================================================#

