import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.svm import LinearSVC


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
# model = Sequential()
# model.add(Dense(100, activation='linear', input_dim=30))    # 기본 activation의 default 값은 'linear'
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))        # relu 는 히든레이어에서만 사용가능함
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1, activation = 'sigmoid'))     # output에서 activatoin = 'sigmoid' ==> 마지막 결과값이 0~1사이로 나옴
                                                # **** 이진 분류모델의 경우 반올림하여 0과 1로 결과값을 받음
model = LinearSVC()

#3. 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam',
#             #   metrics=['accuracy'],
#               metrics=['accuracy', 'mse'])  # 이진 분류함수의 경우 loss = 'binary_crossentropy' 이고 
#                                             # 평가지표 metrics['accuracy']를 사용, 회귀모델의 경우 mse를 사용함

# earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
#                               restore_best_weights=True)
# start_time = time.time() 
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
#                  validation_split=0.2,
#                  callbacks=[earlyStopping],
#                  verbose=1)
# end_time = time.time() - start_time
model.fit(x_train, y_train)


#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)

result = model.score(x_test, y_test)
print('결과 acc : ', result)

# acc = accuracy_score(y_test, y_predict)
# print("=====================================================================")   
# print('acc 스코어 : ', acc)  

# print("===================================================================")   
# print(hist)     
# print("===================================================================")
# print(hist.history)   
# print("=====================================================================")
# print(hist.history['loss'])
# print("=====================================================================")
# print(hist.history['val_loss'])


#================================ SVM 적용 결과 ===================================#
# 결과 acc :  0.9766081871345029
# =================================================================================
# loss :  0.18249185383319855
# mse : 0.05536344274878502
# accuracy :  0.9210526347160339
#==================================================================================#
