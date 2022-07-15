import numpy as np
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

print(x.shape)  # (1797, 64)
x = x.reshape(1797, 64, 1)
print(x.shape)  # (1797, 64, 1)

# One Hot Encoding 
from tensorflow.python.keras.utils.np_utils import to_categorical
y = to_categorical(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)


#2. 모델 구성
model = Sequential()
model.add(LSTM(100, return_sequences=True, 
               activation='linear', input_shape=(64,1)))
model.add(LSTM(100, return_sequences=False, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))


# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중분류에서 loss = 'categorical_crossentropy'를 사용함
              metrics=['accuracy'])   

import datetime
date = datetime.datetime.now()      # 2022-07-07 17:21:42.275191
date = date.strftime("%m%d_%H%M")   # 0707_1723
print(date)

filepath = './_ModelCheckPoint/k26/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=50, mode='min', verbose=1, 
                              restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '07_', date, '_', filename])
                      )

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=10, 
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)       
print('accuracy : ', acc)



#=================================== DNN 모델 ====================================#
# loss :  0.12761995196342468
# accuracy :  0.979629635810852
#=================================================================================#

#=================================== RNN 모델 ====================================#
# loss :  2.0552492141723633
# accuracy :  0.27407407760620117 
#=================================================================================#