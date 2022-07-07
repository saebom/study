import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)


#2. 모델 구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=10))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

# model.save("./_save/keras23_10_save_model1.h5")
# model.save_weights("./_save/keras23_10_save_weights1.h5")

# model = load_model("./_save/keras23_10_save_model1.h5")
# model.load_weights('./_save/keras23_10_save_weights1.h5')
model.load_weights('./_save/keras23_10_save_weights2.h5')

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  

# earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', 
#                               verbose=1, 
#                               restore_best_weights=True)
# start_time = time.time() 
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
#                  validation_split=0.2,
#                  callbacks=[earlyStopping],
#                  verbose=1)
# end_time = time.time() - start_time


# model.save("./_save/keras23_10_save_model2.h5")
# model.save_weights("./_save/keras23_10_save_weights2.h5")
# model = load_model("./_save/keras23_10_save_model2.h5")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  


#===================== save_model1 (random 모델) =================================#
# loss :  [3133.759521484375, 3133.759521484375]
# r2 스코어:  0.4970187135661852
#=================================================================================#

#===================== save_model2 (훈련한 모델) ==================================#
# loss :  [3062.743408203125, 3062.743408203125]
# r2 스코어:  0.508417129644491
#=================================================================================#

#===================== save_weights1 (random 한 가중치 값) ========================#
# loss :  [29056.12109375, 29056.12109375]
# r2 스코어:  -3.663626632334969
#=================================================================================#

#===================== save_weights2 (훈련한 가중치 값) ===========================#
# loss :  [3062.743408203125, 3062.743408203125]
# r2 스코어:  0.508417129644491
#=================================================================================#

