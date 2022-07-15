import numpy as np
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.layers import Conv1D, Flatten, MaxPooling1D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

# One Hot Encoding 
from tensorflow.python.keras.utils.np_utils import to_categorical
y = to_categorical(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (1257, 64) (540, 64) (1257, 10) (540, 10)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(1257, 8*8, 1) 
x_test = x_test.reshape(540, 8*8, 1)
print(x_train.shape)    
print(np.unique(x_train, return_counts=True))


#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=4, padding='same', 
                 activation='relu', input_shape=(8*8, 1)))
model.add(MaxPooling1D(2))  
model.add(Dropout(0.25))     
model.add(Conv1D(64, 4, padding='same', activation='relu'))                
model.add(Dropout(0.25))     
model.add(Conv1D(128, 4, padding='same', activation='relu'))
model.add(Dropout(0.4))     
model.add(Conv1D(128, 4, padding='same', activation='relu'))   
model.add(Dropout(0.25))                 
model.add(Conv1D(64, 4, padding='same', activation='relu'))                
model.add(Dropout(0.2))   
model.add(Conv1D(32, 4, padding='same', activation='relu'))                
model.add(Dropout(0.2))   

model.add(Flatten())   
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.add(Dense(10, activation='softmax'))


# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중분류에서 loss = 'categorical_crossentropy'를 사용함
              metrics=['accuracy'])   

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")   
print(date)

filepath = './_ModelCheckPoint/k41/'
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


# print("=====================================================================")
print("걸린시간 : ", end_time)


#===================================== DNN  ======================================#
# loss :  0.12761995196342468
# accuracy :  0.979629635810852
#=================================================================================#

#==================================== ConvD1 =====================================#
# loss :  1.1689867973327637
# accuracy :  0.8833333253860474
# 걸린시간 :  234.34637236595154 
#=================================================================================#