from concurrent.futures import thread
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
model = XGBRegressor(random_state=72, 
                      n_estimators=1000, 
                      learning_rate = 0.1,
                      max_depth = 6, 
                      gamma= 1,
                    )

#3. 훈련
model.fit(x_train, y_train, early_stopping_rounds=200,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          eval_metric='error',
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
                                   random_state=72, 
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
# r2 :  -3.080452173981257
# 진짜 최종 test 점수 :  -3.080452173981257
# [0.04021428 0.05306398 0.20765993 0.08357305 0.04379774 0.04951727
#  0.05628909 0.05316114 0.34848636 0.06423714]
#==============================================================================#