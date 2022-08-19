# 실습
# 증폭 후 저장한 데이터를 불러와서 
# 완성 및 성능 비교

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(datasets.DESCR)
print(x.shape)  # (581012, 54)

# 스케일링
from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
x = scaler.fit_transform(x)

le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72, stratify=y
    )

############################## pickle 불러오기 // 2. 모델, 3. 훈련 ###############################
import pickle
path = 'D:/study_data/_save/_xg/'
model = pickle.load(open(path+'m45_pickle1_save.dat', 'rb')) # read binary

#4. 평가, 예측
from sklearn.metrics import accuracy_score, f1_score
result = model.score(x_test, y_test)    
print('acc : ', result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("진짜 최종 test 점수 : ", acc)
print('f1_score(macro) : ', f1_score(y_test, y_predict, average='macro'))


#======================= 결과 ===========================#
# 진짜 최종 test 점수 :  0.8952436684078724
# f1_score(macro) :  0.8938595136537548
#========================================================#