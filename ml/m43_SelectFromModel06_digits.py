from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
import numpy as np


#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,)

x = np.delete(x, [0, 3, 4, 6, 8, 11, 12, 13, 14, 16, 22, 23, 24, 
                  31, 32, 39, 40, 41, 42, 44, 47, 48, 49, 57], axis=1)
print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8
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
model = XGBClassifier(random_state=123, 
                      n_estimators=1000, 
                      learning_rate = 0.1,
                      max_depth = 6, 
                      gamma= 1,
                    )

#3. 훈련
model.fit(x_train, y_train, 
          early_stopping_rounds=100,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          eval_metric='merror',
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
    
    selection_model = XGBClassifier(#n_jobs=-1, 
                                   random_state=123, 
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
# acc :  0.9472222222222222
# 진짜 최종 test 점수 :  0.9472222222222222
# [0.         0.01384728 0.01148667 0.00765659 0.00704913 0.04232101
#  0.00774341 0.01089172 0.         0.01051566 0.01776846 0.00481975
#  0.00866377 0.00951625 0.00527508 0.00909332 0.         0.00839456
#  0.00804411 0.0359     0.01121483 0.04519679 0.00442625 0.
#  0.         0.00915349 0.03187055 0.0093536  0.03573615 0.03286256
#  0.02102693 0.         0.         0.05130066 0.00477753 0.0151338
#  0.053122   0.01839486 0.02155397 0.         0.         0.00787295
#  0.03774318 0.04544177 0.01752511 0.01128547 0.0526069  0.
#  0.         0.00308261 0.00727603 0.01454797 0.00673577 0.02488771
#  0.02290419 0.00809022 0.         0.00194705 0.01703124 0.01122978
#  0.05035207 0.01581933 0.04502698 0.01448294]
# =====================================
# (142, 5) (36, 5)
# Thresh=0.073, n=5, Acc:86.11%
# (142, 7) (36, 7)
# Thresh=0.070, n=7, Acc:86.11%
# (142, 13) (36, 13)
# Thresh=0.000, n=13, Acc:97.22%
# (142, 13) (36, 13)
# Thresh=0.000, n=13, Acc:97.22%
# (142, 6) (36, 6)
# Thresh=0.072, n=6, Acc:86.11%
# (142, 9) (36, 9)
# Thresh=0.010, n=9, Acc:97.22%
# (142, 8) (36, 8)
# Thresh=0.055, n=8, Acc:97.22%
# (142, 13) (36, 13)
# Thresh=0.000, n=13, Acc:97.22%
# (142, 13) (36, 13)
# Thresh=0.000, n=13, Acc:97.22%
# (142, 4) (36, 4)
# Thresh=0.100, n=4, Acc:88.89%
# (142, 3) (36, 3)
# Thresh=0.106, n=3, Acc:86.11%
# (142, 1) (36, 1)
# Thresh=0.384, n=1, Acc:72.22%
# (142, 2) (36, 2)
# Thresh=0.131, n=2, Acc:86.11%

#x = np.delete(x, [0, 3, 4, 6, 8, 11, 12, 13, 14, 16, 22, 23, 24, 
#                  31, 32, 39, 40, 41, 42, 44, 47, 48, 49, 57], axis=1)
# acc :  0.9527777777777777
#==============================================================================#