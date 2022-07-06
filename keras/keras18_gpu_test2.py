import numpy as np
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import tensorflow as tf

## cpu와 gpu 테스트

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus):
    print("쥐피유 돈다")
    aaa = 'gpu'
else:
    print("쥐피유 안도라")
    aaa = 'cpu'
    
    
#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
print(datasets.feature_names)

x = datasets['data']         # x = datasets.data 와 동일
y = datasets['target']       # y = datasets.target 와 동일
print(x.shape, y.shape)      # (569, 30) (569,)

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=0
)

#2. 모델 구성
model = Sequential()
model.add(Dense(500, activation='linear', input_dim=30))    # 기본 activation의 default 값은 'linear'
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))        # relu 는 히든레이어에서만 사용가능함
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))     # output에서 activatoin = 'sigmoid' ==> 마지막 결과값이 0~1사이로 나옴
                                                # **** 이진 분류모델의 경우 반올림하여 0과 1로 결과값을 받음
model.summary()

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
            #   metrics=['accuracy'],
              metrics=['accuracy', 'mse'])  # 이진 분류함수의 경우 loss = 'binary_crossentropy' 이고 
                                            # 평가지표 metrics['accuracy']를 사용, 회귀모델의 경우 mse를 사용함

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=200, mode='min', 
                              verbose=1, 
                              restore_best_weights=True)
start_time = time.time() 
hist = model.fit(x_train, y_train, epochs=100, batch_size=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print(aaa, "걸린시간 : ", end_time)  

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 500)               15500
_________________________________________________________________
dense_1 (Dense)              (None, 400)               200400
_________________________________________________________________
dense_2 (Dense)              (None, 300)               120300
_________________________________________________________________
dense_3 (Dense)              (None, 100)               30100
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 101
=================================================================
Total params: 366,401
Trainable params: 366,401
Non-trainable params: 0

'''
# cpu 걸린시간 :  579.6553852558136
# gpu 걸린시간 :  178.88548636436462