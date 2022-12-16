import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager

import tensorflow as tf
tf.random.set_seed(66)  # weight에 난수값 

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

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))     # [1 2 3 4 5 6 7]
# (array([1, 2, 3, 4, 5, 6, 7]), 
#  array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#       dtype=int64))
print(datasets.feature_names)
print(datasets.DESCR)   # 30×30m patches 


# One Hot Encoding 

# print('===================== tensorflow =========================')
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)
# print(y.unique)


# print('===================== pandas =========================')
# import pandas as pd
# # df = pd.DataFrame(y)
# y = pd.get_dummies(y)
# print(y)
# print(y.shape)


print('===================== sklearn =========================')
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
y = y.reshape(-1, 1)
onehot_encoder.fit(y)
y = onehot_encoder.transform(y)

print(x, y)
print(x.shape, y.shape)  

x_train, x_test, y_train, y_test = train_test_split(
   x, y, train_size=0.8, shuffle=True, random_state=72
)

#2. 모델 구성
model = Sequential()
model.add(Dense(500, activation='linear', input_dim=54))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.summary()

#3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중분류에서 loss = 'categorical_crossentropy'를 사용함
              metrics=['accuracy'])   

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=50, mode='min', 
                              verbose=1, 
                              restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=128, 
                 validation_split=0.2,
                #  validation_data=(x_val,y_val),
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)       
print('accuracy : ', acc)

print(aaa, "걸린시간 : ", end_time)


# cpu 걸린시간 :  105.65641951560974
# gpu 걸린시간 :  155.73672604560852


