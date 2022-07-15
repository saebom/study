import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Conv1D, Flatten, Dropout

from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
datasets = load_wine()
x, y = datasets.data, datasets.target

# One Hot Encoding 
from tensorflow.python.keras.utils.np_utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)  # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (124, 13) (54, 13) (124, 3) (54, 3)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(124, 13*1, 1) 
x_test = x_test.reshape(54, 13*1, 1)
print(x_train.shape)    
print(np.unique(x_train, return_counts=True))


#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=1, padding='same', 
                 activation='relu', input_shape=(13*1, 1)))
model.add(Conv1D(64, 1, padding='same', activation='relu'))                
model.add(Dropout(0.25))     
model.add(Conv1D(128, 1, padding='same', activation='relu'))
model.add(Dropout(0.4))     
model.add(Conv1D(128, 1, padding='same', activation='relu'))   
model.add(Dropout(0.25))                 
model.add(Conv1D(64, 1, padding='same', activation='relu'))                
model.add(Dropout(0.2))   
model.add(Conv1D(32, 1, padding='same', activation='relu'))                
model.add(Dropout(0.2))   

model.add(Flatten())   
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))


#3. 훈련
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
                      filepath="".join([filepath, '06_', date, '_', filename])
                      )

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=128, 
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
# loss :  0.0272836834192276
# accuracy :  0.9814814925193787
#=================================================================================#

#==================================== ConvD1 =====================================#
# loss :  0.11987748742103577
# accuracy :  0.9814814925193787
# 걸린시간 :  8.363399267196655
#=================================================================================#
