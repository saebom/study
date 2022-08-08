import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.svm import LinearSVC

import tensorflow as tf
tf.random.set_seed(66)  # weight에 난수값 

#1. 데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)   # 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'

x = datasets['data']
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape)  # (150, 4), (150,)
print('y의 라벨값 : ', np.unique(y))    # y의 라벨값 :  [0 1 2]

# One Hot Encoding 
# from tensorflow.python.keras.utils.np_utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)  # (150, 3)


x_train, x_test, y_train, y_test = train_test_split(
   x, y, train_size=0.8, shuffle=True, random_state=72
)
# print(y_train)
# print(y_test)


#2. 모델 구성
# model = Sequential()
# model.add(Dense(100, activation='linear', input_dim=4))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(3, activation='softmax'))   # 결과값 label이 3개이므로 output 노드의 갯수는 3이 됨 
                                            # softmax를 통해 제일 큰 값이 선택됨(softmax의 값은 전체 합계 1.0이 됨)
                                            # class: 1) Iris-Setosa  2) Iris-Versicolour  3) Iris-Virginica

model = LinearSVC() # epoch량은 모델마다 다름


#3. 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중분류에서 loss = 'categorical_crossentropy'를 사용함
#               metrics=['accuracy'])   

# earlyStopping = EarlyStopping(monitor = 'val_loss', patience=50, mode='min', verbose=1, 
#                               restore_best_weights=True)
# start_time = time.time()

# hist = model.fit(x_train, y_train, epochs=1000, batch_size=10, 
#                  validation_split=0.2,
#                  callbacks=[earlyStopping],
#                  verbose=1)
# end_time = time.time() - start_time
model.fit(x_train, y_train) ################# sklearn에서는 fit에서 compile까지 포함하고 있음

#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss)       
# print('accuracy : ', acc)
# results = model.evaluate(x_test, y_test)
results = model.score(x_test, y_test)   ################## sklearn에서는 evaluate 대신에 score 사용
                                        ################## 분류모델에서는 accuracy_score, 회귀모델에서는 r2_score가 자동으로 연산됨
# print('loss : ', results[0])      
# print('accuracy : ', results[1])
print('결과 acc : ', results)

# print("========================= y_test[:5] =============================")
# print(y_test[:5])
# print("======================== y_predict[:5] ===========================")
# y_predict = model.predict(x_test[:5])  
# print(y_predict)
# print("==================================================================")

# y_test = np.argmax(y_test, axis=1)              # y_predict = y_predict.argmax(axis=1) 와 동일  
# y_predict = np.argmax(y_predict, axis=1)        # y_test = y_test.argmax(axis=1) 와 동일
y_predict = model.predict(x_test)
# y_predict = y_predict.argmax(axis=1)
# y_test = y_test.argmax(axis=1)
print("========================== y_predict =============================")
print(y_predict)
print(y_test)
print("==================================================================")

# acc = accuracy_score(y_test, y_predict)
# print("==================================================================")   
# print('acc 스코어 : ', acc)  

# print("================================================================")   
# print(hist)     
# print("================================================================")
# print(hist.history)     
# print("==================================================================")
# print(hist.history['loss'])
# print("==================================================================")
# print(hist.history['val_loss'])



#===================================== 결  과 ==========================================#
# loss : 0.023589281365275383
# accuracy_score : 1.0
#=======================================================================================#

#=================================== 내 용 정 리 ========================================#
# 1. 다중분류에서는 loss = 'categorical_crossentropy'를 사용
#    결과값 label이 3개이므로 output 노드의 갯수는 3이 됨
#    label(class): 1) Iris-Setosa  2) Iris-Versicolour  3) Iris-Virginica
#    다중분류 모델의 마지막 레이어의 활성화 함수는 softmax 사용
#    => softmax를 통해 제일 큰 값이 선택됨 (softmax의 값은 전체 합계 1.0이 됨)
# 
# 2. One Hot Encoding :: label이 3일 때 [1, 0, 0] or [0, 1, 0] or [0, 0, 1]의 값을 가짐
#    따라서 데이터 전처리 해주어야 함
#  1) python에서 one hot encoding 하기
#    from tensorflow.keras.utils import to_categorical
#    y = to_categorical(y)
#  2) argmax 함수로 1차원 배열에서 가장 큰 값의 인덱스를 찾아서 accuracy_score 값 구하기
#    행(axis=0) 또는 열(axis=1)을 따라 가장 큰 값의 색인을 찾음
#    => y_test와 y_predict에 argmax 함수로 열의 큰 값을(axis=1) 구함
#=======================================================================================#


