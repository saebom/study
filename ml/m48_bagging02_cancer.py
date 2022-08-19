import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#1. 데이터
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) #(569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=72, train_size=0.8, shuffle=True, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from xgboost import XGBClassifier

model = BaggingClassifier(LogisticRegressionCV(), 
# model = BaggingClassifier(LogisticRegression(), 
# model = BaggingClassifier(DecisionTreeClassifier(), 
# model = BaggingClassifier(XGBClassifier(), 
                          n_estimators=100,
                          n_jobs=-1,
                          random_state=72
                          )

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()


#4. 평가, 예측
result = model.score(x_test, y_test)
print('Bagging_XGBClassifier 결과 : ', result)
print('걸린 시간 : ', end - start)


#======================== 결과 ========================#
# 0.9736842105263158
# Bagging_XGBClassifier 결과 :  0.9824561403508771
# 걸린 시간 :  7.494908094406128
# Bagging_LogisticRegressionCV 결과 :  0.9912280701754386
# 걸린 시간 :  8.381080627441406
#======================================================#
