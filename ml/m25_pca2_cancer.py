#맹그러 테스트 [실습]
#30개 컬럼을 몇깨로 했을 때 좋을까??

import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape)  # (569, 30)

pca = PCA(n_components=30)
x = pca.fit_transform(x)
print(x.shape)

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
# PCA 주석처리 전 결과 :  0.8592120537176547
# PCA(n_componets = 2)일 때 : 0.6787089420242385
# PCA(n_componets = 3)일 때 : 0.6885219783819194
# PCA(n_componets = 4)일 때 : 0.8036085817228955
# PCA(n_componets = 5)일 때 : 0.8049715034392401
# PCA(n_componets = 6)일 때 : 0.7856740255486406
# PCA(n_componets = 7)일 때 : 0.8081342286275794
# PCA(n_componets = 8)일 때 : 0.8116852931542745
# PCA(n_componets = 9)일 때 : 0.813119161480511
# PCA(n_componets = 10)일 때 : 0.8087615460203079
# PCA(n_componets = 11)일 때 : 0.8130332787422208
# PCA(n_componets = 12)일 때 : 0.7989522436947265
# PCA(n_componets = 13)일 때 : 0.8138809040288242
# PCA(n_componets = 14)일 때 : 0.8073463478545693
# PCA(n_componets = 15)일 때 : 0.7946357025876187
# PCA(n_componets = 16)일 때 : 0.821625286603341 
# PCA(n_componets = 17)일 때 : 0.8079363249263021
# PCA(n_componets = 18)일 때 : 0.8048557484441533
# PCA(n_componets = 19)일 때 : 0.8148144120537177
# PCA(n_componets = 20)일 때 : 0.8026638716017032
# PCA(n_componets = 21)일 때 : 0.7979963314772356
# PCA(n_componets = 22)일 때 : 0.7985078938748772
# PCA(n_componets = 23)일 때 : 0.7977274811660662
# PCA(n_componets = 24)일 때 : 0.7985937766131674
# PCA(n_componets = 25)일 때 : 0.804463675073698
# PCA(n_componets = 26)일 때 : 0.7994750081886669
# PCA(n_componets = 27)일 때 : 0.8045122174909924
# PCA(n_componets = 28)일 때 : 0.7964989846053063
# PCA(n_componets = 29)일 때 : 0.8020888306583689
# PCA(n_componets = 30)일 때 : 0.7932466426465772
# =====================================================
# ==> PCA(n_componets = 16)일 때 : 0.821625286603341 
#==============================================================================================#