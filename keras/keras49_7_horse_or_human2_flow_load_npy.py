import numpy as np
from keras.preprocessing.image import ImageDataGenerator


#1. 데이터 로드
x_train = np.load('d:/study_data/_save/_npy/keras49_5_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_5_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_5_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_5_test_y.npy')


#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), 
                 input_shape=(150,150,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=30, batch_size=32, 
                 validation_split=0.2,
                 validation_steps=4,
                 )    
end_time = time.time() - start_time

#. 평가, 예측
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
# loss : 0.6876140832901001
# val_loss : 1.0849494934082031
# accuracy : 0.5495495200157166
# val_accuracy : 0.5121951103210449
#================================================================================#


#=============================== 증폭 후 결과값 ===================================#
# loss : 0.6425307989120483
# val_loss : 0.7089158296585083
# accuracy : 0.612500011920929
# val_accuracy : 0.4765625
# =====================================================================
# 걸린시간 :  14.333789110183716
#================================================================================#
