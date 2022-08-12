import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel



#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

x = np.delete(x, [0, 1, 5], axis=1)
print(x.shape)  # (442, 8)

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
# r2 :  0.4399579301416662
# 진짜 최종 test 점수 :  0.4399579301416662
# [0.03877611 0.05315316 0.25949734 0.07350152 0.05215475 0.04736882
#  0.06678308 0.3183003  0.09046488]
# =====================================
# (353, 9) (89, 9)
# Thresh=0.039, n=9, R2:42.42%
# (353, 6) (89, 6)
# Thresh=0.053, n=6, R2:44.94%
# (353, 2) (89, 2)
# Thresh=0.259, n=2, R2:17.42%
# (353, 4) (89, 4)
# Thresh=0.074, n=4, R2:35.66%
# (353, 7) (89, 7)
# Thresh=0.052, n=7, R2:42.00%
# (353, 8) (89, 8)
# Thresh=0.047, n=8, R2:42.39%
# (353, 5) (89, 5)
# Thresh=0.067, n=5, R2:35.25%
# (353, 1) (89, 1)
# Thresh=0.318, n=1, R2:1.38%
# (353, 3) (89, 3)
# Thresh=0.090, n=3, R2:28.76%

# x = np.delete(x, [0, 1, 5], axis=1)
# r2 :  0.49990529572059794
# 진짜 최종 test 점수 :  0.49990529572059794
#==============================================================================#