# 1 357 / 0 212
# 라벨 0을 112개 삭제해서 재구성
# 그런 다음 smote 넣어서 맹글고 넣은거 안넣은거 비교

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (569, 30) (569,)
# print(type(x))          # <class 'numpy.ndarray'>
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357], dtype=int64))

# 라벨 0을 112개 삭제해서 재구성
zero_y = np.where(y==0)
del_y = zero_y[0][100:]
new_x = np.delete(x, del_y, axis=0)
new_y = np.delete(y, del_y)
print(np.unique(new_y, return_counts=True))


x_train, x_test, y_train, y_test = train_test_split(
    new_x, new_y, shuffle=True, random_state=123, train_size=0.8, stratify=new_y
)
print(pd.Series(y_train).value_counts())
# 1    285
# 0     80

print("#========================== SMOTE 적용 후 ============================ ")
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=123, k_neighbors=1)  
x_train, y_train = smote.fit_resample(x_train, y_train)

print(pd.Series(y_train).value_counts())
# 1    285
# 0    285

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = {'n_estimators': [100],
              'learning_rate' : [0.1],
              'max_depth' : [3], #default 6 => 통상 max는 4정도에서 성능이 좋다
              'gamma': [1],
              'min_child_weight': [1],
              'subsample' : [1],
              'colsample_bytree' : [1],
              'colsample_bylevel' : [1],
              'colsample_bynode' : [1],
              'reg_alpha' : [0],
              'reg_lambda' : [1],
              }  

#2. 모델
xgb = XGBClassifier(random_state=123,
                    )
model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
from sklearn.metrics import accuracy_score, f1_score

y_predict = model.predict(x_test) 
score = model.score(x_test, y_test)    
print('acc : ', score)
print('f1_score : ', f1_score(y_test, y_predict))


#======================= 결과 ===========================#
# SMOTE 적용 전 
# acc :  0.9736842105263158
# ====================================
# SMOTE 적용 후
# acc :  0.9736842105263158
# f1_score :  0.9787234042553191
# ====================================
# 0번 인덱스 100개 조정 후 
# acc :  0.9782608695652174
# f1_score :  0.9861111111111112
#========================================================#
