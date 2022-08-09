import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
print(sk.__version__)   # 0.24.2
import warnings
warnings.filterwarnings(action='ignore')


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (506, 13) (506,)


pca = PCA(n_components=5)   # PCA는 주성분분석, 차원축소(열, 컬럼, feature 축소), 
                            # n_components를 통해 주성분을 몇 개로 할 지 결정
x = pca.fit_transform(x)
print(x.shape)  # (506, 2)

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


#===================================== 결과 ===================================================#
# PCA 주석처리 전 결과 :  0.813944143645972
# PCA(n_componets = 11)일 때 : 0.7706232622590622
#==============================================================================================#
# 내용 요약 정리 :: 
# PCA의 본질은 차원 축소임. 
# 차원이 축소됐다는 것은 원본 데이터가 아니라 변환(projection) 된 데이터 == 주성분을 이용해 
# 분석 혹은 모델링을 진행하겠다는 것
#==============================================================================================#
