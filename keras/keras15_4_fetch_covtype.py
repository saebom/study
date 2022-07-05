from bitarray import test
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
model.add(Dense(100, activation='linear', input_dim=54))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중분류에서 loss = 'categorical_crossentropy'를 사용함
              metrics=['accuracy'])   

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=50, mode='min', verbose=1, 
                              restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=5000, batch_size=128, 
                 validation_split=0.2,
                #  validation_data=(x_val,y_val),
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)       
print('accuracy : ', acc)

y_predict = model.predict(x_test)
y_predict = y_predict.argmax(axis=1)      # tensorflow에서 사용 : to_categorical
y_test = y_test.argmax(axis=1)            # tensorflow에서 사용 : to_categorical
# y_predict = tf.argmax(y_predict, axis=1)    # pandas에서 사용 : get_dummies
# y_test = tf.argmax(y_test, axis=1)          # pandas에서 사용 : get_dummies



print("================================ y_predict =================================")
print(y_predict)
print(y_test)
print("============================================================================")

acc = accuracy_score(y_test, y_predict)
print("============================================================================")   
print('acc 스코어 : ', acc)  


import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = 'C:\Windows\Fonts\malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
# plt.title('loss & val_loss')    
plt.title('로스값과 검증로스값')   
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right')  
plt.legend()   
plt.show()



#===================================== 결  과 ==========================================#

#=============================== 1. tensorflow일 때 ====================================# 
# Error 코드 ==> AttributeError: 'numpy.ndarray' object has no attribute 'unique'
# print(y.shape)에서 인덱스가 7개가 아니라 8개로 나타남 ==> (581012, 8)
#=======================================================================================#

#=================================== 2. pandas일 때 ====================================#
# loss : 0.3247310519218445
# accuracy_score : 0.8665696978569031
#=======================================================================================#

#=================================== 3. sklearn일 때 ===================================#
# loss : 0.3247310519218445
# accuracy_score : 0.8665696978569031
#=======================================================================================#

