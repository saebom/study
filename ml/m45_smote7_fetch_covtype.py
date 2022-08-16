# 증폭시켜서 저장하기

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
import time


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

print(pd.Series(y_train).value_counts())
# 1    226640
# 0    169472
# 2     28603
# 6     16408
# 5     13894
# 4      7594
# 3      2198

print("#========================== SMOTE 적용 후 ============================ ")
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=123, k_neighbors=1)
start = time.time()  
x_train, y_train = smote.fit_resample(x_train, y_train)
end = time.time()
print("걸린 시간 : ", end - start) 
print(pd.Series(y_train).value_counts())
# 2    226640
# 0    226640
# 1    226640
# 4    226640
# 6    226640
# 5    226640
# 3    226640

#2. 모델
# model = XGBClassifier()
model = XGBClassifier(random_state=72, 
                      n_estimators=1000, 
                      learning_rate = 0.1,
                      max_depth = 6, 
                      gamma= 1,
                      tree_method='gpu_hist', 
                      predictor='gpu_predictor', 
                      gpu_id=0
                    )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
from sklearn.metrics import accuracy_score, f1_score
result = model.score(x_test, y_test)    
print('acc : ', result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("진짜 최종 test 점수 : ", acc)
print('f1_score(macro) : ', f1_score(y_test, y_predict, average='macro'))

# pickle 저장
import pickle
path = 'D:/study_data/_save/_xg/'
pickle.dump(model, open(path+'m45_pickle1_save.dat', 'wb')) # wright binary

#======================= 결과 ===========================#
# SMOTE 걸린 시간 :  27.997615575790405
# acc :  0.8952436684078724
# f1_score(macro) :  0.8938595136537548
#========================================================#