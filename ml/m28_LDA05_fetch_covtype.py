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
import xgboost as xg
print('xgboost 버전 : ', xg.__version__)    # xgboost 버전 :  1.6.1

'''
01. iris
02. cancer
# 03. diabets
04. wine 
05. fetch_covtype
06. digits
07. kaggle_titanic
'''

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))   
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#       dtype=int64))


# 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

le = LabelEncoder()
y = le.fit_transform(y)

# LDA 
lda = LinearDiscriminantAnalysis(n_components=6)
lda.fit(x, y)
x = lda.transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=172, shuffle=True
)

print(x_train.shape, x_test.shape)  # (120, 2) (30, 2)


#2. 모델
from xgboost import XGBClassifier
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
result = model.score(x_test, y_test)
print('LDA 결과 : ', result)
print('걸린 시간 : ', end - start)


#==================================== 결과 ==================================#
# 적용 전 결과  :  0.9458705885390223
# LDA 적용 결과 :  0.7915544349113189
# 걸린 시간     :  3.518369436264038
#============================================================================#