import numpy as np
import pandas as pd
import os
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, ResNet101
import tensorflow as tf
from PIL import Image


#1. 데이터 로드


data_path = 'D:/study_data/_project1/labeling/'
train = pd.read_csv(data_path + 'paris_train.csv', encoding='cp949')
# print(train.head)
train_img = 'D:/study_data/_project1/img/fashion_img/paris_train/'
test_img = 'D:/study_data/_project1/img/fashion_img/test_img/'

# 이미지 데이터 가져오기
test_images = []

for file in os.listdir(test_img): 
    img_file = file
    test_images.append(img_file) 
# print(len(img_result))  # 2039


# 라벨링 tokenizer    
labels = train['labelId']
# print(labels.head)
token = Tokenizer(filters=',')
token.fit_on_texts(labels)
label_seq = token.texts_to_sequences(labels)
label_length = len(token.word_index) + 1
print(token.word_index)
labels = [np_utils.to_categorical(label, num_classes=label_length, dtype='float32').sum(axis=0)[1:] for label in label_seq]
print(labels)   # 106개

y = np.array(labels[0])
for i in range(1, 1539):
    y = np.vstack((y, labels[i]))
y = np.array(y)
print(y.shape) # (1539, 75)


# validation 라벨링
validation = pd.read_csv(data_path + 'paris_val.csv', encoding='cp949')
# import csv
# with open(data_path + 'paris_val.csv') as cf:
#     data = [list(map(int, i)) for i in csv.reader(cf, delimiter=',')]
    
y_val = validation['labelId']
# print(y_val.head)
tokenizer = Tokenizer(filters = ',')
tokenizer.fit_on_texts(y_val)
label_seq = tokenizer.texts_to_sequences(y_val)
label_length = len(tokenizer.word_index) + 1
print(tokenizer.word_index)

y_val = [np_utils.to_categorical(y_val, num_classes=label_length, dtype='float32').sum(axis=0)[:1] for y_val in label_seq]
val = np.array(y_val[0])
for i in range(1, 341):
    val = np.vstack((val, y_val[i]))
y_val = np.array(val)

z = np.zeros((341, 1))
y_val = np.append(y_val, z, axis = 1)
print(y_val.shape)  # (341, 2)

#2. Image DataGenerator
train_datagen = ImageDataGenerator(
    rescale = 1./255,       
    horizontal_flip= True,  
    vertical_flip=True,    
    width_shift_range=0.1,  
    height_shift_range=0.1, 
    rotation_range=5,       
    zoom_range=1.2,         
    shear_range=0.7,        
    fill_mode='nearest',     
    featurewise_center=True,
    featurewise_std_normalization=True,  
    )

train_datagen2 = ImageDataGenerator(
    rescale = 1./255,           
    )

# train image
import tqdm
from tensorflow.keras.utils import load_img, img_to_array
train_image = []
for i in tqdm.tqdm(range(train.shape[0])):
    img = load_img(train_img + str(i+1) + '.jpg', target_size=(50, 60, 3))
    # img = load_img(train_img + str(i+1) + '.jpg', target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img/255
    train_image.append(img)
x = np.array(train_image)
print(x.shape) # (3227, 50, 60, 3), (2298, 50, 60, 3), (1830, 50, 60, 3), (1539, 50, 60, 3)

#test image
test_data = train_datagen2.flow_from_directory(
    'D:/study_data/_project1/img/fashion_img/test_img',
    target_size=(50, 60)
)
# test_data = np.array(test_images, dtype='float32') / 255.0
    
# np.save('d:/study_data/_save/_npy/project1_newyork_x.npy', arr = x)
# np.save('d:/study_data/_save/_npy/project1_newyork_y.npy', arr=y)
# np.save('d:/study_data/_save/_npy/project1_newyork_xval.npy', arr = x_val)
# np.save('d:/study_data/_save/_npy/project1_newyork_yval.npy', arr= y_val)
np.save('D:/study_data/_save/_npy/project1_test1.npy', arr = test_data[0][0])
np.save('D:/study_data/_save/_npy/project1_test2.npy', arr = test_data[0][1])

# x = np.load('D:/study_data/_save/_npy/project1_newyork_x.npy')
# y = np.load('D:/study_data/_save/_npy/project1_newyork_y.npy')
# x_val = np.load('D:/study_data/_save/_npy/project1_newyork_xval.npy')
# y_val = np.load('D:/study_data/_save/_npy/project1_newyork_yval.npy')


#2. 모델
'''
from keras.applications import ResNet50, ResNet101, ResNet50V2, ResNet152V2

model = ResNet101(include_top=True, weights=None, input_shape=(50, 60, 3), 
                 pooling=max, classes=75)
model = ResNet50(include_top=True, weights=None, input_shape=(50, 60, 3), 
                 pooling=max, classes=75)
model = ResNet152V2(include_top=True, weights=None, input_shape=(50, 60, 3), 
                 pooling=max, classes=75)
model = ResNet50(include_top=True, weights=None, input_shape=(50, 60, 3), 
                 pooling=max, classes=75)



from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), 
                 input_shape=(50,60,3), activation='relu'))
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


model.summary()


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import time
start_time = time.time()
hist = hist = model.fit(x, y, epochs=10, batch_size=32, 
                 validation_data=(x_val, y_val))     
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
'''

# ======================================= loss 및 accuracy =========================================
# loss : 0.6876140832901001
# val_loss : 1.0849494934082031
# accuracy : 0.5495495200157166
# val_accuracy : 0.5121951103210449
# ==================================================================================================
