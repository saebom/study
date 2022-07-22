import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import DirectoryIterator
from keras.layers import MaxPooling2D, Dropout
from sympy import Max
import tensorboard 

#1. 데이터 로드

x_train = np.load('d:/study_data/_save/_npy/keras49_5_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_5_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_5_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_5_test_y.npy')


#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2, 2), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))  
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))  
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=30, batch_size=32, 
                 validation_split=0.2, verbose=1)   
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
# loss : 0.047354985028505325
# val_loss : 2.5059962272644043
# accuracy : 0.9800124764442444
# val_accuracy : 0.5459088087081909
#================================================================================#


#============================ save_load 후 결과값 ================================#
# loss : 0.32446855306625366
# val_loss : 0.5323543548583984
# accuracy : 0.8818125128746033
# val_accuracy : 0.7936000227928162
# =====================================================================
# 걸린시간 :  1719.3597767353058
#================================================================================#
