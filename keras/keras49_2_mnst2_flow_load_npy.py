from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


#1. 데이터 로드
x_train = np.load('d:/study_data/_save/_npy/keras49_2_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_2_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_2_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_2_test_y.npy')


#2. 모델링 
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4, 4),    
                 padding='same', 
                 input_shape=(28, 28, 1)))      
model.add(MaxPooling2D(2, 2))           
model.add(Dropout(0.2))
model.add(Conv2D(64, (4, 4), padding='valid', activation='relu'))                
model.add(MaxPooling2D(2, 2))          
model.add(Dropout(0.2))
model.add(Conv2D(64, (4, 4), padding='same', activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())    # (N, 63)  (N, 175)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
print(date)

filepath = './_ModelCheckPoint/k28/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '01_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=50, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

print("=====================================================================")
print("걸린시간 : ", end_time)


#=============================== 이전 결과값 ====================================#
# loss :  0.04272077605128288
# accuracy :  0.9915000200271606
#================================================================================#


#=============================== 증폭 후 결과값 ===================================#
# loss :  0.03503519669175148
# accuracy :  0.9919999837875366
# =====================================================================
# 걸린시간 :  558.1001296043396
#================================================================================#
