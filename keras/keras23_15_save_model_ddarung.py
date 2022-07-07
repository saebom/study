import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping
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
    x, y, train_size=0.7, random_state=66
)


#2. 모델구성
model = Sequential()
model.add(Dense(100, activation = 'linear', input_dim=9))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

# model.save("./_save/keras23_15_save_model1.h5")
# model.save_weights("./_save/keras23_15_save_weights1.h5")

# model = load_model("./_save/keras23_15_save_model1.h5")
# model.load_weights('./_save/keras23_15_save_weights1.h5')
model.load_weights('./_save/keras23_15_save_weights2.h5')


#3. 컴파일, 훈련
model.compile(loss = 'mae', optimizer="adam", metrics=['mse'])

# earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='min',
#                               restore_best_weights=True,
#                               verbose=1)
# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=5000, batch_size=100, 
#           validation_split=0.2,
#           callbacks=[earlyStopping],
#           verbose=1)
# end_time = time.time() - start_time

# model.save("./_save/keras23_15_save_model2.h5")
# model.save_weights("./_save/keras23_15_save_weights2.h5")
# model = load_model("./_save/keras23_15_save_model2.h5")

#4. 평가예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)  

#### mse를 rmse로 변환 ####
y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


#===================== save_model1 (random 모델) =================================#
# loss :  [39.9486083984375, 3185.638427734375]
# R2 :  0.487477611950784
#=================================================================================#

#===================== save_model2 (훈련한 모델) ==================================#
# loss :  [30.945289611816406, 2030.0643310546875]
# R2 :  0.6733925181954212
#=================================================================================#

#===================== save_weights1 (random 한 가중치 값) ========================#
# loss :  [76.99070739746094, 10363.2470703125]
# R2 :  -0.667294002141523
#=================================================================================#

#===================== save_weights2 (훈련한 가중치 값) ===========================#
# loss :  [30.945289611816406, 2030.0643310546875]
# R2 :  0.6733925181954212
#=================================================================================#


