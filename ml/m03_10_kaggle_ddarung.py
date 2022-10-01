import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv',   # 예측에서 사용!!
                       index_col=0)

#### 결측치 처리 ####
test_set = test_set.fillna(method='ffill')
train_set = train_set.dropna()  # nan 값 삭제

x = train_set.drop(['count'], axis=1)
y = train_set['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=72
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = LinearSVR()


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)  
results = model.score(x_test, y_test)
print('결과 r2 : ', results)

# #### mse를 rmse로 변환 ####
# y_predict = model.predict(x_test)
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))

# rmse = RMSE(y_test, y_predict)
# print("RMSE : ", rmse)

# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2)

# y_summit = model.predict(test_set)
# print(y_summit)
# print(y_summit.shape)   # (715, 1)

# submission = pd.read_csv(path + 'submission.csv')
# submission['count'] = y_summit
# submission.to_csv(path + 'submission1.csv', index=False)


# print("=================================================================")   
# print(hist)     # <tensorflow.python.keras.callbacks.History object at 0x000002664FF27AF0>
# print("=================================================================")
# print(hist.history)     # loss 와 val_loss의 key, value를 합쳐놓은 것
# print("=================================================================")
# print(hist.history['loss'])
# print("=================================================================")
# print(hist.history['val_loss'])


#===================================== 결  과 ==========================================#
# LinearSVR() 결과 r2 :  
# LinearRegression() 결과 r2 :  
# KNeighborsRegressor() 결과 r2 :  
# DecisionTreeRegressor() 결과 결과 r2 : 
# RandomForestRegressor() 결과 r2 : 
#=======================================================================================#

