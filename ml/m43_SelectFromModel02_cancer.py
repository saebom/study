from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
import numpy as np


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

x = np.delete(x, [5, 10, 11, 12, 22, 23], axis=1)
print(x.shape)  # (569, 29)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y
)

parameters = {'n_estimators': [100],
              'learning_rate' : [0.1],
              'max_depth' : [3], 
              'gamma': [1],
              'min_child_weight': [1],
              'subsample' : [1],
              'colsample_bytree' : [1],
              'colsample_bylevel' : [1],
              'colsample_bynode' : [1],
              'reg_alpha' : [0],
              'reg_lambda' : [1]
              }  

#2. 모델
model = XGBClassifier(random_state=72, 
                      n_estimators=1000, 
                      learning_rate = 0.1,
                      max_depth = 6, 
                      gamma= 1,
                    )

#3. 훈련
model.fit(x_train, y_train, early_stopping_rounds=100,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          eval_metric='error',
          )


#4. 평가, 예측
result = model.score(x_test, y_test)    
print('acc : ', result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("진짜 최종 test 점수 : ", acc)

print(model.feature_importances_)

thresholds = model.feature_importances_
print("=====================================")
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)   
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBClassifier(n_jobs=-1, 
                                   random_state=72, 
                                   n_estimators=1000, 
                                   learning_rate = 0.1,
                                   max_depth = 6, 
                                   gamma= 1,)
    
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, Acc:%.2f%%"
          %(thresh, select_x_train.shape[1], score*100))


#=================================== 결과 =====================================#
# acc :  0.9736842105263158
# 진짜 최종 test 점수 :  0.9736842105263158
# [0.00846497 0.02200363 0.         0.01753137 0.01272222 0.01100117
#  0.05361047 0.04635817 0.         0.00649525 0.01141741 0.
#  0.00900095 0.01222006 0.00943855 0.00790024 0.01214139 0.02233558
#  0.         0.00903257 0.01388057 0.02094848 0.2847385  0.27382424
#  0.014983   0.00853961 0.01743228 0.07514252 0.01108644 0.00775036]

#  0.00855463 0.01746293 0.07527463 0.01110593 0.00776399]
# =====================================
# (455, 7) (114, 7)
# Thresh=0.022, n=7, Acc:94.74%
# (455, 24) (114, 24)
# Thresh=0.008, n=24, Acc:97.37%
# (455, 10) (114, 10)
# Thresh=0.016, n=10, Acc:95.61%
# (455, 13) (114, 13)
# Thresh=0.013, n=13, Acc:97.37%
# (455, 18) (114, 18)
# Thresh=0.011, n=18, Acc:97.37%
# (455, 4) (114, 4)
# Thresh=0.054, n=4, Acc:95.61%
# (455, 5) (114, 5)
# Thresh=0.046, n=5, Acc:94.74%
# (455, 29) (114, 29)
# Thresh=0.000, n=29, Acc:97.37%
# (455, 26) (114, 26)
# Thresh=0.007, n=26, Acc:97.37%
# (455, 16) (114, 16)
# Thresh=0.011, n=16, Acc:98.25%
# (455, 29) (114, 29)
# Thresh=0.000, n=29, Acc:97.37%
# (455, 21) (114, 21)
# Thresh=0.009, n=21, Acc:98.25%
# (455, 14) (114, 14)
# Thresh=0.012, n=14, Acc:98.25%
# (455, 19) (114, 19)
# Thresh=0.009, n=19, Acc:96.49%
# (455, 23) (114, 23)
# Thresh=0.008, n=23, Acc:97.37%
# (455, 15) (114, 15)
# Thresh=0.012, n=15, Acc:98.25%
# (455, 6) (114, 6)
# Thresh=0.022, n=6, Acc:94.74%
# (455, 29) (114, 29)
# Thresh=0.000, n=29, Acc:97.37%
# (455, 20) (114, 20)
# Thresh=0.009, n=20, Acc:98.25%
# (455, 12) (114, 12)
# Thresh=0.014, n=12, Acc:96.49%
# (455, 8) (114, 8)
# Thresh=0.021, n=8, Acc:96.49%
# (455, 1) (114, 1)
# Thresh=0.285, n=1, Acc:90.35%
# (455, 2) (114, 2)
# Thresh=0.274, n=2, Acc:90.35%
# (455, 11) (114, 11)
# Thresh=0.015, n=11, Acc:95.61%
# (455, 22) (114, 22)
# Thresh=0.009, n=22, Acc:96.49%
# (455, 9) (114, 9)
# Thresh=0.017, n=9, Acc:96.49%
# (455, 3) (114, 3)
# Thresh=0.075, n=3, Acc:95.61%
# (455, 17) (114, 17)
# Thresh=0.011, n=17, Acc:98.25%
# (455, 25) (114, 25)
# Thresh=0.008, n=25, Acc:96.49%

# x = np.delete(x, [0, 4, 5, 10, 11, 12, 22, 23], axis=1)
# acc :  0.9912280701754386
# 진짜 최종 test 점수 :  0.9912280701754386
#==============================================================================#