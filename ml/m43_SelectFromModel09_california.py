import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel



#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

x = np.delete(x, [3, 4], axis=1)
print(x.shape)  # (20640, 6)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=72, train_size=0.8
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
model = XGBRegressor(random_state=123, 
                      n_estimators=1000, 
                      learning_rate = 0.1,
                      max_depth = 6, 
                      gamma= 1,
                    )

#3. 훈련
model.fit(x_train, y_train, early_stopping_rounds=200,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          eval_metric='rmse',
          )


#4. 평가, 예측
result = model.score(x_test, y_test)    
print('r2 : ', result)

y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)
print("진짜 최종 test 점수 : ", acc)

print(model.feature_importances_)

thresholds = model.feature_importances_
print("=====================================")
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)   #prefit이면 feature_importances 상의 자신보다 작은 것이 반환됨
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBRegressor(n_jobs=-1, 
                                   random_state=123, 
                                   n_estimators=1000, 
                                   learning_rate = 0.1,
                                   max_depth = 6, 
                                   gamma= 1,)
    
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, R2:%.2f%%"
          %(thresh, select_x_train.shape[1], score*100))


#=================================== 결과 =====================================#
# r2 :  0.8418246494845028
# 진짜 최종 test 점수 :  0.8418246494845028
# [0.5561829  0.07589684 0.03784238 0.02116891 0.02123137 0.1403118
#  0.06779255 0.07957323]
# =====================================
# (16512, 1) (4128, 1)
# Thresh=0.556, n=1, R2:49.46%
# (16512, 4) (4128, 4)
# Thresh=0.076, n=4, R2:74.59%
# (16512, 6) (4128, 6)
# Thresh=0.038, n=6, R2:84.37%
# (16512, 8) (4128, 8)
# Thresh=0.021, n=8, R2:84.18%
# (16512, 7) (4128, 7)
# Thresh=0.021, n=7, R2:84.12%
# (16512, 2) (4128, 2)
# Thresh=0.140, n=2, R2:58.48%
# (16512, 5) (4128, 5)
# Thresh=0.068, n=5, R2:83.48%
# (16512, 3) (4128, 3)
# Thresh=0.080, n=3, R2:72.32%

# x = np.delete(x, [3, 4], axis=1)
# r2 :  0.8437110641109735
# 진짜 최종 test 점수 :  0.8437110641109735
#==============================================================================#