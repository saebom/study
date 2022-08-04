# import Numpy, Scipy, and Matplotlib
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import os
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import PredefinedSplit, train_test_split
from sympy import continued_fraction_reduce
import tensorflow as tf
from PIL import Image


#1. 데이터 로드

x = np.load('d:/study_data/_save/_npy/project1_total_x.npy')
y = np.load('d:/study_data/_save/_npy/project1_total_y.npy')
x_val = np.load('d:/study_data/_save/_npy/project1_total_xval.npy')
y_val = np.load('d:/study_data/_save/_npy/project1_total_yval.npy')


data_path = 'D:/study_data/_project1/labeling/'
test = pd.read_csv(data_path + 'paris_val.csv', encoding='cp949')
test_img = 'D:/study_data/_project1/img/fashion_img/paris_val/'
columns = ['ImageId', 'years', 'season', 'region', 'designer', 'labelId']


for col in columns:
    print(col)
    print(test[col].unique())
    
value_counts = test['labelId'].value_counts()
indexes = value_counts.index
values = value_counts.values
for i in range(len(value_counts)):
    if values[i] < 1000:
        break
    
     
# 이미지 데이터 가져오기
img_result = []

for file in os.listdir(test_img): 
    img_file = file
    img_result.append(img_file) 
print(len(img_result))  # 161


# 라벨 데이터 가져오기
labels = []
   
#used_columns = ['region', 'labelId']
used_columns = ['labelId']

for index, row in test.iterrows():
    if row['ImageId'] in img_result:
        continued_fraction_reduce
    tags = []
    
    for col in used_columns:
        tags.append(row[col])
        
    labels.append(tags)

import tqdm
from tensorflow.keras.utils import load_img, img_to_array
test_image = []
for i in tqdm.tqdm(range(test.shape[0])):
    img = load_img(test_img + str(i+1) + '.jpg', target_size=(50, 60, 3))
    img = img_to_array(img)
    img = img/255
    test_image.append(img)
x = np.array(test_image)
print(x.shape) # (8894, 50, 60, 3)

    
# Image DataGenerator
test_data = np.array(test_image, dtype='float32') / 255.0
test_labels = np.array(labels)
print(test_data.shape)      # (776, 50, 60, 3)
print(test_labels.shape)    # (776, 1)

test_data = np.load('d:/study_data/_save/_npy/project1_test_x.npy')
test_labels = np.load('d:/study_data/_save/_npy/project1_test_y.npy')
test_image = np.save('d:/study_data/_save/_npy/project1_total_y.npy', arr=test_image)

# import Keras's functional api
from keras.models import Model, Sequential

model = Sequential()
# get the weights from the last layer
gap_weights = model.layers[-1].get_weights()[0]

# create a new model to output the feature maps and the predicted labels
cam_model = Model(inputs=model.input, 
                    outputs=(model.layers[-3].output, model.layers[-1].output)) 

# make the prediction for a set of test images
features, results = cam_model.predict(test_img)

# check the prediction for 10 test images
for idx in range(10):   
    # get the feature map of the test image
    features_for_one_img = features[idx, :, :, :]

    # map the feature map to the original size
    height_roomout = test_img_size_h / features_for_one_img.shape[0]
    width_roomout = test_img_size_w / features_for_one_img.shape[1]
    cam_features = sp.ndimage.zoom(features_for_one_img, (height_roomout, width_roomout, 1), order=2)
        
    # get the predicted label with the maximum probability
    pred = np.argmax(results[idx])
    
    # prepare the final display
    plt.figure(facecolor='white')
    
    # get the weights of class activation map
    cam_weights = gap_weights[:, pred]

    # create the class activation map
    cam_output = np.dot(cam_features, cam_weights)
    
    # draw the class activation map
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    buf = 'Predicted Class = ' + fashion_name[pred] + ', Probability = ' + str(results[idx][pred])
    plt.xlabel(buf)
    plt.imshow(t_pic[idx], alpha=0.5)
    plt.imshow(cam_output, cmap='jet', alpha=0.5)
     
    plt.show()  
    
