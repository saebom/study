import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
print(sk.__version__)   # 0.24.2
import warnings
warnings.filterwarnings(action='ignore')


#1. 데이터
# datasets = load_boston()
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)


pca = PCA(n_components=30)   # PCA는 주성분분석, 차원축소(열, 컬럼, feature 축소), 
                             # n_components를 통해 주성분을 몇 개로 할 지 결정
x = pca.fit_transform(x)
print(x.shape)  # (506, 2)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
# [8.05823175e-01 1.63051968e-01 2.13486092e-02 6.95699061e-03
#  1.29995193e-03 7.27220158e-04 4.19044539e-04 2.48538539e-04
#  8.53912023e-05 3.08071548e-05 6.65623182e-06]
print(sum(pca_EVR)) # 0.999998352533973

cumsum = np.cumsum(pca_EVR) # cumsum은 누적합
print(cumsum)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()


"""
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72, shuffle=True
)

#2. 모델
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train) #, eval_metric='error')

#4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 : ', result)

"""

#===================================== 결과 ===================================================#
# PCA 주석처리 전 결과 :  0.813944143645972
# PCA(n_componets = 11)일 때 : 0.7706232622590622
#==============================================================================================#

