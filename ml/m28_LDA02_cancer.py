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
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  # (569, 30) (569,)

# 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

# LDA 
# lda = LinearDiscriminantAnalysis()
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(x, y)
x = lda.transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, shuffle=True
)

print(x_train.shape, x_test.shape)  # (455, 1) (114, 1)
print(np.unique(y_train, return_counts=True))   
# (array([0, 1]), array([169, 286], dtype=int64))

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
print('LDA 적용 결과 : ', result)
print('걸린 시간 : ', end - start)


#==================================== 결과 ==================================#
# 적용 전 결과  : 0.9736842105263158
# LDA 적용 결과 : 0.9824561403508771
# 걸린 시간     : 0.3585796356201172
#============================================================================#