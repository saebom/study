import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.layers import Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

# One Hot Encoding 
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
y = y.reshape(-1, 1)
onehot_encoder.fit(y)
y = onehot_encoder.transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (406708, 54) (174304, 54) (406708, 7) (174304, 7)


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(406708, 3, 6, 3) 
x_test = x_test.reshape(174304, 3, 6, 3)
print(x_train.shape)    
print(np.unique(x_train, return_counts=True))


#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', 
                 activation='relu', input_shape=(3, 6, 3)))
model.add(Dropout(0.25))     
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))                
model.add(Dropout(0.25))     
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(Dropout(0.4))     
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))   
model.add(Dropout(0.25))                 
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))                
model.add(Dropout(0.2))   
model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))                
model.add(Dropout(0.2))   

model.add(Flatten())   
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))


#3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중분류에서 loss = 'categorical_crossentropy'를 사용함
              metrics=['accuracy'])   

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")   

filepath = './_ModelCheckPoint/k24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=50, mode='min', verbose=1, 
                              restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '14_', date, '_', filename])
                      )

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=256, 
                 validation_split=0.2,
                #  validation_data=(x_val,y_val),
                 callbacks=[earlyStopping, mcp],
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


#================================= 1. 기본 출력 ===================================#
# loss :  0.12761995196342468
# accuracy :  0.979629635810852
# 07_0707_1954_0033-0.1017.hdf5
#=================================================================================#