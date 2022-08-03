import numpy as np
import tensorflow as tf
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm


#1. 데이터

data_path = 'D:/study_data/_project1/labeling/'
train = pd.read_csv(data_path + 'paris_train.csv', encoding='cp949')
train_img = 'D:/study_data/_project1/img/fashion_img/paris_train/'

# Reading the training images
train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('train/'+str(train['id'][i])+'.png', target_size=(50,60,3), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
x = np.array(train_image)

# Creating the target variable
y=train['labelId'].values
y = to_categorical(y)

# Creating validation set
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

# Define the model structure
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
# Training the model
model.fit(X_train, y_train, epochs=12, validation_data=(X_test, y_test))


test_file = pd.read_csv('test.csv')
test_image = []
for i in tqdm(range(test_file.shape[0])):
    img = image.load_img('test/'+str(test_file['id'][i])+'.png', target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)
prediction = model.predict_classes(test)

# To Download/import the sample_submission given:
download = drive.CreateFile({'id': '1CtEbaEukaKD_WcChk2MqCVM_dSFLliTr'})
download.GetContentFile('sample_submission.csv')
sample = pd.read_csv('sample_submission.csv')
sample['id'] = test_file['id']
sample['label'] = prediction
sample.to_csv('sample.csv', header=True, index=False)

# to download the resulted file locally
from google.colab import files
files.download( 'sample.csv' )