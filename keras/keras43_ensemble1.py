#1. 데이터
from lightgbm import train
import numpy as np

x1_datasets = np.array([range(100), range(301, 401)])   # 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]) # 원유, 돈육, 밀 가격
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)

print(x1.shape, x2.shape)       # (100, 2) (100, 3)

y = np.array(range(2001, 2101)) # 금리  (100, )

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1, x2, y, train_size=0.7, shuffle=True, random_state=32   
)


# print(x1_train, x1_test, x2_train, x2_test, y_train, y_test)
print(x1_train.shape, x1_test.shape)    # (70, 2) (30, 2) 
print(x2_train.shape, x2_test.shape)    # (70, 3) (30, 3) 
print(y_train.shape, y_test.shape)      # (70,) (30,)

#2. 모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(128, activation='relu', name='sb1')(input1)
dense2 = Dense(128, activation='relu', name='sb2')(dense1)
dense3 = Dense(64, activation='relu', name='sb3')(dense2)
output1 = Dense(10, activation='relu', name='out_sb1')(dense3)

#2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(32, activation='relu', name='sb11')(input2)
dense12 = Dense(32, activation='relu', name='sb12')(dense11)
dense13 = Dense(16, activation='relu', name='sb13')(dense12)
dense14 = Dense(14, activation='relu', name='sb14')(dense13)
output2 = Dense(10, activation='relu', name='out_sb2')(dense14)

from tensorflow.python.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(2, activation='relu', name='mg2')(merge1)
merge3 = Dense(3, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

# 모델 정의
model = Model(inputs=[input1, input2], outputs=last_output)
model.summary()

# [실습]
#3. 컴파일, 훈련
model.compile(loss = 'mae', optimizer='adam', metrics=['mse'])

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
print(date)

filepath = './_ModelCheckPoint/k43/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
earlyStopping = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '01_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit([x1_train, x2_train], y_train, epochs=200, batch_size=64,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss : ', loss)

from sklearn.metrics import r2_score
import tensorflow as tf
y_predict = model.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


# loss :  [108.43125915527344, 15712.177734375]
# loss :  [70.96102142333984, 6753.26025390625]
# loss :  [4.2182536125183105, 18.704668045043945]
# loss :  [1.6370849609375, 2.918449878692627]

#=================================== ensemble ====================================#
# loss :  [22.840328216552734, 699.7650756835938]
# R2 :  0.15713519847770352
#=================================================================================#

# Model: "model"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_2 (InputLayer)            [(None, 3)]          0
# __________________________________________________________________________________________________
# input_1 (InputLayer)            [(None, 2)]          0
# __________________________________________________________________________________________________
# sb11 (Dense)                    (None, 11)           44          input_2[0][0]
# __________________________________________________________________________________________________
# sb1 (Dense)                     (None, 1)            3           input_1[0][0]
# __________________________________________________________________________________________________
# sb12 (Dense)                    (None, 12)           144         sb11[0][0]
# __________________________________________________________________________________________________
# sb2 (Dense)                     (None, 2)            4           sb1[0][0]
# __________________________________________________________________________________________________
# sb13 (Dense)                    (None, 13)           169         sb12[0][0]
# __________________________________________________________________________________________________
# sb3 (Dense)                     (None, 3)            9           sb2[0][0]
# __________________________________________________________________________________________________
# sb14 (Dense)                    (None, 14)           196         sb13[0][0]
# __________________________________________________________________________________________________
# out_sb1 (Dense)                 (None, 10)           40          sb3[0][0]
# __________________________________________________________________________________________________
# out_sb2 (Dense)                 (None, 10)           150         sb14[0][0]
# __________________________________________________________________________________________________
# mg1 (Concatenate)               (None, 20)           0           out_sb1[0][0]
#                                                                  out_sb2[0][0]
# __________________________________________________________________________________________________
# mg2 (Dense)                     (None, 2)            42          mg1[0][0]
# __________________________________________________________________________________________________
# dense (Dense)                   (None, 3)            9           mg2[0][0]
# __________________________________________________________________________________________________
# last (Dense)                    (None, 1)            4           dense[0][0]
# ==================================================================================================
# Total params: 814
# Trainable params: 814
# Non-trainable params: 0

