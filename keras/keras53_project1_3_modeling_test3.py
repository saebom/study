import numpy as np
import pandas as pd
import os
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
import tensorflow as tf
from PIL import Image


#1. 데이터 로드
data_path = 'D:/study_data/_project1/labeling/'
# train = pd.read_csv(data_path + 'milan_train.csv', encoding='cp949')
train = pd.read_csv(data_path + 'milan_train.csv')
print(train.head)
train_img = 'D:/study_data/_project1/img/fashion_img/milan_train/'
val_img = 'D:/study_data/_project1/img/fashion_img/milan_val/'

# 이미지 데이터 가져오기
img_result = []

for file in os.listdir(train_img): 
    img_file = file
    img_result.append(img_file) 
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
for i in range(1, 2039):
    y = np.vstack((y, labels[i]))
y = np.array(y)

# validation set
# validation = pd.read_csv(data_path + 'milan_val.csv', encoding='cp949')
validation = pd.read_csv(data_path + 'milan_val.csv')
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
for i in range(1, 150):
    val = np.vstack((val, y_val[i]))
y_val = np.array(val)

z = np.zeros((150, 1))
y_val = np.append(y_val, z, axis = 1)
print(y_val.shape)  # (150, 2)

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
    img = img_to_array(img)
    img = img/255
    train_image.append(img)
x = np.array(train_image)
print(x.shape) # (3227, 50, 60, 3), (2298, 50, 60, 3)

#validation image
val_image = []
for i in tqdm.tqdm(range(validation.shape[0])):
    img = load_img(val_img + str(i+1) + '.jpg', target_size=(50, 60, 3))
    img = img_to_array(img)
    img = img/255
    val_image.append(img)
x_val = np.array(val_image)
print(x_val.shape) # (776, 50, 60, 3), (558, 50, 60, 3)
    
    
np.save('d:/study_data/_save/_npy/project1_train.npy', arr = x)
np.save('d:/study_data/_save/_npy/project1-validation.npy', arr=x_val)


#2. 모델

# model = ResNet50(include_top=True, weights='imagenet', input_shape=(150, 150, 3), 
#                  pooling=max, classes=1000)
model = ResNet50(include_top=True, weights='imagenet', input_shape=(150, 150, 3), 
                 pooling=max, classes=1000)
model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(x, y, epochs=10, batch_size=32, 
                 validation_data=(x_val, y_val))   
# hist = model.fit_generator(train_generator, epochs = 20, steps_per_epoch = 1, 
#                     validation_data = validation_generator,
#                     validation_steps = 4,
#                     ) 

#. 평가, 예측
accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss :', loss[-1])       
print('val_loss :', val_loss[-1])
print('accuracy :', accuracy[-1])
print('val_accuracy :', val_accuracy[-1])

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


# ==================================== fit loss 및 accuracy ========================================
# loss : 0.6876140832901001
# val_loss : 1.0849494934082031
# accuracy : 0.5495495200157166
# val_accuracy : 0.5121951103210449
# ==================================================================================================
