import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetLarge, NASNetMobile
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from keras.applications import Xception
import tensorflow as tf


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)


# 2. 모델
models = [
          # VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2,
          # ResNet152, ResNet152V2, DenseNet121, DenseNet169, DenseNet201, 
          # InceptionV3, InceptionResNetV2, 
          MobileNet, MobileNetV2,
          MobileNetV3Small, MobileNetV3Large, 
          NASNetLarge, NASNetMobile,
          EfficientNetB0, EfficientNetB1, EfficientNetB7,
          Xception]

for i in models:
    tfModel = i(weights='imagenet', include_top=False,
                input_shape=(32, 32, 3))

    model = Sequential()
    model.add(tfModel)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.trainable = False
    # model.summary()


    print(" ==================================================== ")
    print("모델명 : ", i.__name__)
    print("전체 가중치 갯수 : ", len(model.weights))
    print("훈련 가능 가중치 갯수 : ", len(model.trainable_weights))
      

########################### 결과출력 ####################################
# 모델명 :  ResNet50
# 전체 가중치 갯수 :  322
# 훈련 가능 가중치 갯수 :  0
#  ==================================================== 
# 모델명 :  ResNet50V2
# 전체 가중치 갯수 :  274
# 훈련 가능 가중치 갯수 :  0
#  ==================================================== 
# 모델명 :  ResNet101
# 전체 가중치 갯수 :  628
# 훈련 가능 가중치 갯수 :  0
#  ==================================================== 
# 모델명 :  ResNet101V2
# 전체 가중치 갯수 :  546
# 훈련 가능 가중치 갯수 :  0
#  ==================================================== 
# 모델명 :  ResNet152
# 전체 가중치 갯수 :  934
# 훈련 가능 가중치 갯수 :  0
#  ==================================================== 
# 모델명 :  ResNet152V2
# 전체 가중치 갯수 :  818
# 훈련 가능 가중치 갯수 :  0
#  ==================================================== 
# 모델명 :  DenseNet121
# 전체 가중치 갯수 :  608
# 훈련 가능 가중치 갯수 :  0
#  ==================================================== 
# 모델명 :  DenseNet169
# 전체 가중치 갯수 :  848
# 훈련 가능 가중치 갯수 :  0
#  ==================================================== 
# 모델명 :  DenseNet201
# 전체 가중치 갯수 :  1008
# 훈련 가능 가중치 갯수 :  0
#  ==================================================== 
# 모델명 :  InceptionV3
# ValueError: Input size must be at least 75x75; Received: input_shape=(32, 32, 3)
# ===================================================== 
# 모델명 :  InceptionResNetV2
# ValueError: Input size must be at least 75x75; Received: input_shape=(32, 32, 3)
#  ==================================================== 
# 모델명 :  MobileNet
# 전체 가중치 갯수 :  139
# 훈련 가능 가중치 갯수 :  0
#  ==================================================== 
# 모델명 :  MobileNetV2
# 전체 가중치 갯수 :  264
# 훈련 가능 가중치 갯수 :  0
#  ==================================================== 
# 모델명 :  MobileNetV3Small
# 전체 가중치 갯수 :  210
# 훈련 가능 가중치 갯수 :  0
#  ==================================================== 
# 모델명 :  MobileNetV3Large
# 전체 가중치 갯수 :  266
# 훈련 가능 가중치 갯수 :  0
#  ==================================================== 
# 모델명 :  NASNetLarge
# 전체 가중치 갯수 :  1548
# 훈련 가능 가중치 갯수 :  0
#  ==================================================== 
# 모델명 :  NASNetMobile
# 전체 가중치 갯수 :  1128
# 훈련 가능 가중치 갯수 :  0
#  ==================================================== 
# 모델명 :  EfficientNetB0
# 전체 가중치 갯수 :  316
# 훈련 가능 가중치 갯수 :  0
# ==================================================== 
# 모델명 :  EfficientNetB1
# 전체 가중치 갯수 :  444
# 훈련 가능 가중치 갯수 :  0
#  ==================================================== 
# 모델명 :  EfficientNetB7
# 전체 가중치 갯수 :  1042
# 훈련 가능 가중치 갯수 :  0