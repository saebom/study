from tabnanny import verbose
import numpy as np
import pandas as pd
import os
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, ResNet101, ResNet50V2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image


#1. 데이터 로드

x_train = np.load('D:/study_data/_save/_npy/project1_paris_x.npy')
y_train = np.load('D:/study_data/_save/_npy/project1_paris_y.npy')
x_test = np.load('D:/study_data/_save/_npy/project1_paris_xval.npy')
y_test = np.load('D:/study_data/_save/_npy/project1_paris_yval.npy')


#2. 모델

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Activation

# VGGNet16
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), strides=1,
                 input_shape=(70,80,3), activation='relu'))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(128, kernel_size=(3,3), strides=1,
                 activation='relu'))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(256, kernel_size=(3,3), strides=1,
                 activation='relu'))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Conv2D(256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Conv2D(256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(512, kernel_size=(3,3), strides=1,
                 activation='relu'))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Conv2D(512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Conv2D(512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))


model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(150, activation='softmax'))


model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import time
start_time = time.time()
hist = hist = model.fit(x_train, y_train, epochs=100, batch_size=128,
                #  validation_data=(x_test, y_test),
                   validation_split=0.2)     
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


# ======================================= loss 및 accuracy =========================================
# loss : 0.6876140832901001
# val_loss : 1.0849494934082031
# accuracy : 0.5495495200157166
# val_accuracy : 0.5121951103210449
# ==================================================================================================
