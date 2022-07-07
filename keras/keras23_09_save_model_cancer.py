import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=30))    # 기본 activation의 default 값은 'linear'
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))        # relu 는 히든레이어에서만 사용가능함
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))     # output에서 activatoin = 'sigmoid' ==> 마지막 결과값이 0~1사이로 나옴
                                                # **** 이진 분류모델의 경우 반올림하여 0과 1로 결과값을 받음

# model.save("./_save/keras23_9_save_model1.h5")
# model.save_weights("./_save/keras23_9_save_weights1.h5")

# model = load_model("./_save/keras23_9_save_model1.h5")
# model.load_weights('./_save/keras23_9_save_weights1.h5')
model.load_weights('./_save/keras23_9_save_weights2.h5')

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
            #   metrics=['accuracy'],
              metrics=['accuracy', 'mse'])  # 이진 분류함수의 경우 loss = 'binary_crossentropy' 이고 
                                            # 평가지표 metrics['accuracy']를 사용, 회귀모델의 경우 mse를 사용함

# earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
#                               restore_best_weights=True)
# start_time = time.time() 
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
#                  validation_split=0.2,
#                  callbacks=[earlyStopping],
#                  verbose=1)
# end_time = time.time() - start_time

# model.save("./_save/keras23_9_save_model2.h5")
# model.save_weights("./_save/keras23_9_save_weights2.h5")

# model = load_model("./_save/keras23_9_save_model2.h5")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

#=== 과제 1. accuracy_score 완성하기 =========================================
y_predict = model.predict(x_test)

y_predict = y_predict.flatten()                 # 차원 펴주기
y_predict = np.where(y_predict > 0.5, 1 , 0)   # 0.5보다 크면 1, 작으면 0
print(y_predict)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
# ====================================================== 과제 1 끝 ===========

acc = accuracy_score(y_test, y_predict)
print("=====================================================================")   
print('acc 스코어 : ', acc)  


#===================== save_model1 (random 모델) =================================#
# loss :  [0.0695854127407074, 0.019719084724783897]
# acc 스코어 :  0.9707602339181286
#=================================================================================#

#===================== save_model2 (훈련한 모델) ==================================#
# loss :  [0.11271049827337265, 0.026913585141301155]
# acc 스코어 :  0.9532163742690059
#=================================================================================#

#===================== save_weights1 (random 한 가중치 값) ========================#
# loss :  [0.7123346924781799, 0.25951674580574036]
# acc 스코어 :  0.3391812865497076
#=================================================================================#

#===================== save_weights2 (훈련한 가중치 값) ===========================#
# loss :  [0.11271049827337265, 0.026913585141301155]
# acc 스코어 :  0.9532163742690059
#=================================================================================#
