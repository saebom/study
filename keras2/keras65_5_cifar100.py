# [실습] trainable = True, False 비교해가면서 만들어서 결과 비교

import tensorflow as tf
import numpy as np
from keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)


# 2. 모델
vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))

# vgg16.summary()
vgg16.trainable = False     # 가중치를 동결시킨다!!! vgg16의 trainable을 시키지 않는다
# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

# model.trainable = False     # 모델의 trainable을 시키지 않는다
model.summary()
 
print(len(model.weights))
print(len(model.trainable_weights))

############################################################
import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
############################################################

#3. 컴파일, 훈련
model.compile(optimizer='adam', metrics=['accuracy'], 
                loss='sparse_categorical_crossentropy')

model.fit(x_train, y_train, epochs=300, validation_split=0.4, verbose=1,
          batch_size=128)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

# ================================================================ 결과 ======================================================================================= #
# #                                                                                             Layer Type                Layer Name  Layer Trainable
# 0  <keras.engine.functional.Functional object at 0x000001E39FA4D790>                                    vgg16                     True
# 1  <keras.layers.pooling.global_average_pooling2d.GlobalAveragePooling2D object at 0x000001E39FA46220>  global_average_pooling2d  True
# 2  <keras.layers.core.dense.Dense object at 0x000001E39FAA9550>                                         dense                     True
# 3  <keras.layers.core.dense.Dense object at 0x000001E39FA33370>                                         dense_1                   True
# ============================================================================================================================================================== #
# vgg16.trainable = False 
# model.trainable = True
# loss :  32.53131866455078
# acc :  0.22130000591278076