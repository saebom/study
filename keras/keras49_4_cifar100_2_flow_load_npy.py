from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.datasets import cifar100
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from keras.preprocessing.image import ImageDataGenerator

#1. 데이터 로드

x_train = np.load('d:/study_data/_save/_npy/keras49_4_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_4_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_4_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_4_test_y.npy')


#2. 모델링 
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', 
                 activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(2, 2))  
model.add(Dropout(0.25))     
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))                
model.add(MaxPooling2D(2, 2))   
model.add(Dropout(0.25))     
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(2, 2))     
model.add(Dropout(0.4))     
model.add(Conv2D(254, (3, 3), padding='same', activation='relu'))   
model.add(MaxPooling2D(2, 2))     
model.add(Dropout(0.4))                 

model.add(Flatten())   
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='Nadam', 
              metrics=['accuracy'])

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
print(date)

filepath = './_ModelCheckPoint/k28/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '03_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=300, batch_size=128,
                 validation_split=0.3,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss :', loss[-1])       
print('val_loss :', val_loss[-1])
print('accuracy :', accuracy[-1])
print('val_accuracy :', val_accuracy[-1])

print("=====================================================================")
print("걸린시간 : ", end_time)


#그래프로 비교
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path = 'C:\Windows\Fonts\malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.plot(hist.history['accuracy'], marker='.', c='orange', label='accuracy')
plt.plot(hist.history['val_accuracy'], marker='.', c='green', label='val_accuracy')
plt.grid()    
plt.title('로스값과 검증로스값')    
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()   
plt.show()


#=============================== 이전 결과값 ====================================#
# loss :  1.872557282447815
# accuracy :  0.5112000107765198
#================================================================================#


#=============================== 증폭 후 결과값 ===================================#
# loss : 2.977832078933716
# val_loss : 3.685940742492676
# accuracy : 0.2620200216770172
# val_accuracy : 0.1534757912158966
# =====================================================================
# 걸린시간 :  2187.8850173950195
#================================================================================#
