from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])


#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.summary() # 연산량을 보여 줌 Layer-OutputShape-Param 

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 500)               27500
_________________________________________________________________
dense_1 (Dense)              (None, 400)               200400
_________________________________________________________________
dense_2 (Dense)              (None, 300)               120300
_________________________________________________________________
dense_3 (Dense)              (None, 200)               60200
_________________________________________________________________
dense_4 (Dense)              (None, 100)               20100
_________________________________________________________________
dense_5 (Dense)              (None, 100)               10100
_________________________________________________________________
dense_6 (Dense)              (None, 7)                 707
=================================================================
Total params: 439,307
Trainable params: 439,307
Non-trainable params: 0

'''
