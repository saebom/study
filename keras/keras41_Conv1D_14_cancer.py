import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (398, 30) (171, 30) (398,) (171,)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(398, 6*5, 1) 
x_test = x_test.reshape(171, 6*5, 1)
print(x_train.shape)    
print(np.unique(x_train, return_counts=True))


#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, padding='same', 
                 activation='relu', input_shape=(6*5, 1)))
model.add(MaxPooling1D(2))  
model.add(Dropout(0.25))     
model.add(Conv1D(64, 3, padding='same', activation='relu'))                
model.add(MaxPooling1D(2))  
model.add(Dropout(0.25))     
model.add(Conv1D(128, 3, padding='same', activation='relu'))
model.add(Dropout(0.4))     
model.add(Conv1D(128, 3, padding='same', activation='relu'))   
model.add(Dropout(0.25))                 
model.add(Conv1D(64, 3, padding='same', activation='relu'))                
model.add(Dropout(0.2))   
model.add(Conv1D(32, 3, padding='same', activation='relu'))                
model.add(Dropout(0.2))   

model.add(Flatten())   
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
            #   metrics=['accuracy'],
              metrics=['accuracy', 'mse'])  # 이진 분류함수의 경우 loss = 'binary_crossentropy' 이고 
                                            # 평가지표 metrics['accuracy']를 사용, 회귀모델의 경우 mse를 사용함

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M") 
print(date)

filepath = './_ModelCheckPoint/k41/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '04_', date, '_', filename])
                      )
start_time = time.time() 
hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

#=== 과제 1. accuracy_score 완성하기 =========================================
y_predict = model.predict(x_test)

y_predict = y_predict.flatten()                 # 차원 펴주기
y_predict = np.where(y_predict > 0.5, 1 , 0)   # 0.5보다 크면 1, 작으면 0
print(y_predict)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
# ====================================================== 과제 1 끝 ===========

acc = accuracy_score(y_test, y_predict)
print("=====================================================================")   
print('acc 스코어 : ', acc)  


print("=====================================================================")
print("걸린시간 : ", end_time)


#====================================== DNN ========================================#
# loss :  [0.07133284211158752, 0.019666554406285286]
# acc 스코어 :  0.9707602339181286
# 걸린시간 :  
#===================================================================================#

#==================================== Conv1D =======================================#
# loss :  [0.4060918688774109, 1.804222583770752]
# acc 스코어 :  0.9532163742690059
# 걸린시간 : 37.61471486091614
#===================================================================================#

