from tkinter import N
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.datasets import load_wine, load_digits
from sklearn.datasets import load_boston
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
# datasets = load_iris()  # (150, 4) -> (150, 2)
# datasets = load_breast_cancer() # (569, 30) -> (569, 1)
# datasets = load_wine() # (178, 13) -> (178, 2)
# datasets = fetch_covtype()  # (581012, 54) -> (581012, 6)
datasets = load_digits()    # (1797, 64) -> (1797, 9)

x = datasets.data
y = datasets.target
print(x.shape)  

# LDA 
lda = LinearDiscriminantAnalysis()
# lda = LinearDiscriminantAnalysis(n_components=6)
lda.fit(x, y)
x = lda.transform(x)
print(x.shape)  

lda_EVR = lda.explained_variance_ratio_ # explained_variance_ratio_는 
                                        # 각각의 주성분 벡터가 이루는 축에 투영(projection)한 결과의 
                                        # 분산 비율(각 eigenvalue의 비율과 같은 의미)
print(lda_EVR)

cumsum = np.cumsum(lda_EVR) # cumsum은 누적합
print(cumsum)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()
