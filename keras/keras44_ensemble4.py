#1. 데이터
import numpy as np

x1_datasets = np.array([range(100), range(301, 401)])   # 삼성전자 종가, 하이닉스 종가
x1 = np.transpose(x1_datasets)

print(x1.shape)       # (100, 2) 

y1 = np.array(range(2001, 2101)) # 금리  (100, )
y2 = np.array(range(201, 301)) # 작년 금리  (100, )



# [실습] 만들기

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test \
= train_test_split(
    x1, y1, y2, train_size=0.7, shuffle=True, random_state=32   
)

# print(x1_train, x1_test, x2_train, x2_test, y_train, y_test)
print(x1_train.shape, x1_test.shape)    # (70, 2) (30, 2) 
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

#2-2. output 모델1
output41 = Dense(10)(output1)
output42 = Dense(10)(output41)
output2 = Dense(1)(output42)

#2-3. output 모델2
output51 = Dense(10)(output1)
output52 = Dense(10)(output51)
output3 = Dense(1)(output52)


# 모델 정의
model = Model(inputs=input1, outputs=[output2, output3])
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
hist = model.fit(x1_train, [y1_train, y2_train], epochs=500, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x1_test, [y1_test, y2_test])
print('loss : ', loss)


from sklearn.metrics import r2_score
y_predict = model.predict(x1_test)

r2_1 = r2_score(y1_test, y_predict[0])
r2_2 = r2_score(y2_test, y_predict[1])

print("R2(y1) : ", r2_1)
print("R2(y2) : ", r2_1)

# loss1, loss2 = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
# print('loss : ', loss1)
# print('loss : ', loss2)

# from sklearn.metrics import r2_score

# y_predict = model.predict([x1_test, x2_test, x3_test])

# r, r2_1, r2_2 = r2_score([y1_test, y2_test], [y_predict[0], y_predict[1]])
# print("R2 : ", r)
# print("R2_1 : ", r2_1)
# print("R2_2 : ", r2_2)


#================================================= ensemble ==============================================#
# loss :  [28.724531173706055, 3.7662556171417236, 24.958276748657227, 19.967693328857422, 833.3328247070312]
# R2(y1) :  0.9759489752402094
# R2(y2) :  0.9759489752402094
#==========================================================================================================#

