from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=72, train_size=0.8, stratify=y
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
model.fit(x_train, y_train, early_stopping_rounds=200,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          eval_metric='error',
          )


#4. 평가, 예측
result = model.score(x_test, y_test)    
print('acc : ', result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("진짜 최종 test 점수 : ", acc)

# import pickle
path = 'D:/study_data/_save/_xg/'
# pickle.dump(model, open(path+'m39_pickle1_save.dat', 'wb')) 

# import joblib
# joblib.dump(model, path+'m40_joblib1_save.dat')

model.save_model(path + 'm41_xgb1_save_model.dat')


#=================================== 결과 =====================================#
# acc :  0.9912280701754386
# 진짜 최종 test 점수 :  0.9912280701754386
#==============================================================================#