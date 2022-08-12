import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel



#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

x = np.delete(x, [1, 3, 6, 11], axis=1)
print(x.shape)  # (506, 10)

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
# r2 :  0.8171126669797995
# 진짜 최종 test 점수 :  0.8171126669797995
# [0.03447016 0.00450792 0.01357801 0.00690147 0.04257242 0.33320126
#  0.00766984 0.03262796 0.02582668 0.01882491 0.0167862  0.00820649
#  0.45482668]
# =====================================
# (404, 4) (102, 4)
# Thresh=0.034, n=4, R2:76.26%
# (404, 13) (102, 13)
# Thresh=0.005, n=13, R2:81.57%
# (404, 9) (102, 9)
# Thresh=0.014, n=9, R2:84.10%
# (404, 12) (102, 12)
# Thresh=0.007, n=12, R2:81.57%
# (404, 3) (102, 3)
# Thresh=0.043, n=3, R2:77.50%
# (404, 2) (102, 2)
# Thresh=0.333, n=2, R2:55.82%
# (404, 11) (102, 11)
# Thresh=0.008, n=11, R2:80.95%
# (404, 5) (102, 5)
# Thresh=0.033, n=5, R2:80.80%
# (404, 6) (102, 6)
# Thresh=0.026, n=6, R2:82.05%
# (404, 7) (102, 7)
# Thresh=0.019, n=7, R2:80.79%
# (404, 8) (102, 8)
# Thresh=0.017, n=8, R2:82.35%
# (404, 10) (102, 10)
# Thresh=0.008, n=10, R2:84.10%
# (404, 1) (102, 1)
# Thresh=0.455, n=1, R2:32.91%

# x = np.delete(x, [1, 3, 6, 11], axis=1)
# r2 :  0.8412775482727702
# 진짜 최종 test 점수 :  0.8412775482727702
#==============================================================================#