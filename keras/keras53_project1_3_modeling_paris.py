from tabnanny import verbose
import numpy as np
import pandas as pd
import os
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, ResNet101, ResNet50V2, ResNet152V2
from sklearn.model_selection import train_test_split
from sympy import continued_fraction_reduce
import tensorflow as tf
from PIL import Image


#1. 데이터 로드

data_path = 'D:/study_data/_project1/labeling/'
train = pd.read_csv(data_path + 'paris_train.csv', encoding='cp949')
train_img = 'D:/study_data/_project1/img/fashion_img/paris_train/'

print(train.head(5))
columns = ['ImageId', 'years', 'season', 'region', 'designer', 'labelId']

for col in columns:
    print(col)
    print(train[col].unique())
    
value_counts = train['labelId'].value_counts()
indexes = value_counts.index
values = value_counts.values

     
# 이미지 데이터 가져오기
img_result = []

for file in os.listdir(train_img): 
    img_file = file
    img_result.append(img_file) 
# print(len(img_result))  # 2039


# 라벨링 tokenizer 
labels = []
   
used_columns = ['region', 'labelId']

for index, row in train.iterrows():
    if row['Imageid'] in img_result:
        continued_fraction_reduce
    tags = []
    
    for col in used_columns:
        tags.append(row[col])
        
    labels.append(tags)
    
    
# print(labels.head)
token = Tokenizer(filters=',')
token.fit_on_texts(labels)
label_seq = token.texts_to_sequences(labels)
label_length = len(token.word_index) + 1
print(token.word_index)
labels = [np_utils.to_categorical(label, num_classes=label_length, dtype='float32').sum(axis=0)[1:] for label in label_seq]
print(labels)   # 150개

y = np.array(labels[0])
for i in range(1, 3227):
    y = np.vstack((y, labels[i]))
y = np.array(y)
print(y.shape) # (3227, 150)


# Image DataGenerator

import tqdm
from tensorflow.keras.utils import load_img, img_to_array
train_image = []
for i in tqdm.tqdm(range(train.shape[0])):
    img = load_img(train_img + str(i+1) + '.jpg', target_size=(70, 80, 3))
    # img = load_img(train_img + str(i+1) + '.jpg', target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img/255
    train_image.append(img)
x = np.array(train_image)
print(x.shape) # (3227, 50, 60, 3), (2298, 50, 60, 3), (1830, 50, 60, 3), (1539, 50, 60, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=False)   
    
np.save('d:/study_data/_save/_npy/project1_paris_x.npy', arr = x_train)
np.save('d:/study_data/_save/_npy/project1_paris_y.npy', arr= y_train)
np.save('d:/study_data/_save/_npy/project1_paris_xval.npy', arr = x_test)
np.save('d:/study_data/_save/_npy/project1_paris_yval.npy', arr= y_test)

# x_train = np.load('D:/study_data/_save/_npy/project1_paris_x.npy')
# y_train = np.load('D:/study_data/_save/_npy/project1_paris_y.npy')
# x_test = np.load('D:/study_data/_save/_npy/project1_paris_xval.npy')
# y_test = np.load('D:/study_data/_save/_npy/project1_paris_yval.npy')


#2. 모델

# model = ResNet101(include_top=True, weights=None, input_shape=(50, 60, 3), 
#                  pooling=max, classes=150)
# model = ResNet50(include_top=True, weights=None, input_shape=(70, 80, 3), 
#                  pooling=max, classes=150)
model = ResNet152V2(include_top=True, weights=None, input_shape=(70, 80, 3), 
                 pooling=max, classes=157)


''' ,  
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), 
                 input_shape=(224,224,3), activation='relu'))
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
model.add(Dense(150, activation='softmax'))

'''
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

