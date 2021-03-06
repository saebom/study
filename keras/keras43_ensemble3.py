#1. 데이터
import numpy as np

x1_datasets = np.array([range(100), range(301, 401)])   # 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]) # 원유, 돈육, 밀 가격
x3_datasets = np.array([range(100, 200), range(1301, 1401)])   # 우리반 아이큐, 우리반 키 
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)
x3 = np.transpose(x3_datasets)

print(x1.shape, x2.shape, x3.shape)       # (100, 2) (100, 3) (100, 2)

y1 = np.array(range(2001, 2101)) # 금리  (100, )
y2 = np.array(range(201, 301)) # 작년 금리  (100, )



# [실습] 만들기

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, x2, x3, y1, y2, train_size=0.7, shuffle=True, random_state=32   
)

# print(x1_train, x1_test, x2_train, x2_test, y_train, y_test)
print(x1_train.shape, x1_test.shape)    # (70, 2) (30, 2) 
print(x2_train.shape, x2_test.shape)    # (70, 3) (30, 3) 
print(x3_train.shape, x3_test.shape)    # (70, 2) (30, 2)
print(y1_train.shape, y1_test.shape)      # (70,) (30,)
print(y2_train.shape, y2_test.shape)      # (70,) (30,)

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

#2-3. 모델3
input3 = Input(shape=(2,))
dense21 = Dense(128, activation='relu', name='sb21')(input3)
dense22 = Dense(128, activation='relu', name='sb22')(dense21)
dense23 = Dense(64, activation='relu', name='sb23')(dense22)
output3 = Dense(10, activation='relu', name='out_sb3')(dense23)


from tensorflow.python.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2, output3], name='mg1')
merge2 = Dense(2, activation='relu', name='mg2')(merge1)
merge3 = Dense(3, name='mg3')(merge2)
last_output1 = Dense(1, name='last1')(merge3)
# last_output2 = Dense(1, name='last2')(merge3)

#2-4. y1 모델
merge2 = Dense(64, activation='relu', name='mg2')(merge1)
merge3 = Dense(64, name='mg3')(merge2)
last_output2 = Dense(1, name='last1')(merge3)

#2-5. y2 모델
merge4 = Dense(64, activation='relu', name='mg4')(merge1)
merge5 = Dense(32, name='mg5')(merge4)
last_output3 = Dense(1, name='last2')(merge5)

# 모델 정의
model = Model(inputs=[input1, input2, input3], outputs=[last_output2, last_output3])
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
hist = model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=500, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss_1 = model.evaluate([x1_test, x2_test, x3_test], y1_test)
loss_2 = model.evaluate([x1_test, x2_test, x3_test], y2_test)
print('loss(y1) : ', loss_1)
print('loss(y2) : ', loss_2)

from sklearn.metrics import r2_score

y1_predict, y2_predict = model.predict([x1_test, x2_test, x3_test])

r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
print("R2(y1) : ", r2_1)
print("R2(y2) : ", r2_2)



#================================================= ensemble ==============================================#
#               loss - last_loss - last_mse - val_loss - val_last_loss 
# loss(y1) :  [1802.688232421875, 2.30194091796875, 1800.38623046875, 9.270050048828125, 3241391.25]
# loss(y2) :  [1801.4617919921875, 1800.8804931640625, 0.5812454223632812, 3243179.0, 0.5378387570381165]
# R2(y1) :  0.9888342545052683
# R2(y2) :  0.9993521749400164
#==========================================================================================================#

