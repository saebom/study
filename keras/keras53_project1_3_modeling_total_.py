from tabnanny import verbose
import numpy as np
import pandas as pd
import os
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, ResNet101, ResNet50V2, ResNet152V2
from sklearn.model_selection import PredefinedSplit, train_test_split
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from sympy import continued_fraction_reduce
import tensorflow as tf
from PIL import Image


#1. 데이터 로드
data_path = 'D:/study_data/_project1/labeling/'
train = pd.read_csv(data_path + 'total_train.csv', encoding='cp949')
# train = pd.read_csv(data_path + 'london_train.csv', encoding='cp949')
train_img = 'D:/study_data/_project1/img/fashion_img/total_img/'
# train_img = 'D:/study_data/_project1/img/fashion_img/london_train/'
columns = ['ImageId', 'years', 'season', 'region', 'designer', 'labelId']
test1 = np.load('D:/study_data/_save/_npy/project1_test1.npy')
test2 = np.load('D:/study_data/_save/_npy/project1_test2.npy')


for col in columns:
    print(col)
    print(train[col].unique())
    
value_counts = train['labelId'].value_counts()
indexes = value_counts.index
values = value_counts.values
for i in range(len(value_counts)):
    if values[i] < 1000:
        break
    
     
# 이미지 데이터 가져오기
img_result = []

for file in os.listdir(train_img): 
    img_file = file
    img_result.append(img_file) 
print(len(img_result))  # 161


# 라벨 데이터 가져오기
labels = []
   
# used_columns = ['region', 'labelId']
used_columns = ['labelId']

for index, row in train.iterrows():
    if row['ImageId'] in img_result:
        continued_fraction_reduce
    tags = []
    
    for col in used_columns:
        tags.append(row[col])
        
    labels.append(tags)

import tqdm
from tensorflow.keras.utils import load_img, img_to_array
train_image = []
for i in tqdm.tqdm(range(train.shape[0])):
    img = load_img(train_img + str(i+1) + '.jpg', target_size=(50, 60, 3))
    img = img_to_array(img)
    img = img/255
    train_image.append(img)
x = np.array(train_image)
print(x.shape) # (8894, 50, 60, 3)

    
# Image DataGenerator
data = np.array(train_image, dtype='float32') / 255.0
labels = np.array(labels)
    
# multilabelbinarizer
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

print('===================================')
# print(mlb.classes_)
# print(len(mlb.classes_)) # 165
# print(labels[0])
    
x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.8, shuffle=False)

# np.save('d:/study_data/_save/_npy/project1_total_x.npy', arr = x_train)
# np.save('d:/study_data/_save/_npy/project1_total_y.npy', arr=y_train)
# np.save('d:/study_data/_save/_npy/project1_total_xval.npy', arr = x_test)
# np.save('d:/study_data/_save/_npy/project1_total_yval.npy', arr= y_test)

# x_train = np.load('d:/study_data/_save/_npy/project1_total_x.npy')
# y_train = np.load('d:/study_data/_save/_npy/project1_total_y.npy')
# x_test = np.load('d:/study_data/_save/_npy/project1_total_xval.npy')
# y_test = np.load('d:/study_data/_save/_npy/project1_total_yval.npy')


#2. 모델
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), 
                 input_shape=(50, 60, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(4,4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(4,4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('sigmoid'))
out = len(mlb.classes_)
model.add(Dense(out))
model.add(Activation('softmax'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

filepath = './_ModelCheckPoint/k53/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
import datetime
date = datetime.datetime.now()     
date = date.strftime("%m%d_%H%M")   
print(date)

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '01_', date, '_', filename])
                      )

import time
start_time = time.time()
hist = hist = model.fit(x_train, y_train, epochs=50, batch_size=128,
                        validation_split=0.2,
                        verbose=1)     
end_time = time.time() - start_time


# #3.#load_model
# model = load_model('./_ModelCheckPoint/01_0803_1747_0001-6.5613.hdf5')


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



# 테스트 데이터 로드

predict = model.predict(test1)
predict[0]
np.argmax(predict[0])



# ======================================= loss 및 accuracy =========================================
# loss : 1.2357330322265625
# val_loss : 8.140939712524414
# accuracy : 0.7136628031730652
# val_accuracy : 0.06576402485370636
# ==================================================================================================

