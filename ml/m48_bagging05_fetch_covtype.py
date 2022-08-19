from tkinter import N
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.datasets import load_wine, load_digits
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(datasets.DESCR)
print(x.shape)  # (581012, 54)

# 라벨인코딩
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72, stratify=y
    )

# 스케일링
from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
# model = BaggingClassifier(LogisticRegression(), 
# model = BaggingClassifier(DecisionTreeClassifier(), 
model = BaggingClassifier(XGBClassifier(), 
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
# Bagging_DecisionTreeClassifier 결과 :  
# 걸린 시간 :  
#======================================================#
