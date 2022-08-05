import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
tf.random.set_seed(777)

#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target


# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, random_state=72
# )

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)


#2. 모델 구성
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()


#3.4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)
print('ACC : ', scores, '\n cross_val_score : ', round(np.mean(scores), 4))


#=============================== StraitifiedKFold 적용 결과 ==================================================#
# ACC :  [0.34873267 0.50268848 0.33609064 0.54225716 0.43462749] 
#  cross_val_score :  0.4329
#==================================== KFold 적용 결과 ========================================================#
# ACC :  [0.27802401 0.34003731 0.39098519 0.48544607 0.49050933 0.50800739
#  0.35756718 0.5360902  0.45670569]
#  cross_val_score :  0.427
#=============================================================================================================#
