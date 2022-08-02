# 넘파이에서 불러와서 모델 구성
# 성능 비교 
# 증폭해서 npy에 저장
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

#1. 데이터 로드

x_train = np.load('D:/study_data/_save/_npy/project1_paris_x.npy')
y_train = np.load('D:/study_data/_save/_npy/project1_paris_y.npy')
x_test = np.load('D:/study_data/_save/_npy/project1_paris_xval.npy')
y_test = np.load('D:/study_data/_save/_npy/project1_paris_yval.npy')


#2. 모델
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import *

model = Sequential()
input1 = Input(shape=(70, 80, 1))
dense1 = Conv2D(64, kernel_size=(4, 4), padding='same')(input1)
dense2 = MaxPooling2D(2, 2)(dense1)
dense3 = Dropout(0.2)(dense2)
dense4 = Conv2D(64, (4, 4), padding='valid', activation='relu')(dense3)
dense5 = MaxPooling2D(2, 2)(dense4)         
dense6 = Dropout(0.2)(dense5)
dense7 = Conv2D(64, (4, 4), padding='same', activation='relu')(dense6)
dense8 = MaxPooling2D(2, 2)(dense7)         
dense9 = Dropout(0.2)(dense8)
dense10 = Flatten()(dense9)    # (N, 63)  (N, 175)
dense11 = Dense(128, activation='relu')(dense10)
dense12 = Dropout(0.2)(dense11)
dense13 = Dense(64, activation='relu')(dense12)
output1= Dense(150, activation='softmax')(dense13)
model = Model(inputs=input1, outputs=output1)
model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse','accuracy'])

import time
start_time = time.time()
# hist = model.fit_generator(xy_train, epochs=100, steps_per_epoch=len(xy_train))
hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
                 validation_split=0.2)
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
# loss :  0.28520235419273376
# accuracy :  0.9143000245094299
#================================================================================#


#=============================== 증폭 후 결과값 ===================================#
# loss : 0.32446855306625366
# val_loss : 0.5323543548583984
# accuracy : 0.8818125128746033
# val_accuracy : 0.7936000227928162
# =====================================================================
# 걸린시간 :  1719.3597767353058
#================================================================================#
