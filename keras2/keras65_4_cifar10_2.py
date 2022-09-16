# [실습] trainable = True, False 비교해가면서 만들어서 결과 비교

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.applications import VGG16
from sklearn.metrics import accuracy_score
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

# loss의 스케일 조정을 위해 0 ~ 255 -> 0 ~ 1 범위로 만들어줌
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

mean = np.mean(x_train, axis=(0 , 1 , 2 , 3))
std = np.std(x_train, axis=(0 , 1 , 2 , 3))
x_train = (x_train-mean)/std
x_test = (x_test-mean)/std

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)
print(x_train.shape)    # (50000, 32, 32, 3)
print(np.unique(x_train, return_counts=True))

# # One Hot Encoding
# from tensorflow.python.keras.utils.np_utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape)


# 2. 모델
vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))

# vgg16.summary()
# vgg16.trainable = False     # 가중치를 동결시킨다!!! vgg16의 trainable을 시키지 않는다
# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.summary()

model.trainable = False 
print(len(model.weights))
print(len(model.trainable_weights))

############################################################
import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
#############################################################

#3. 컴파일, 훈련
model.compile(optimizer='adam', metrics=['accuracy'], 
                loss='sparse_categorical_crossentropy')

model.fit(x_train, y_train, epochs=300, validation_split=0.4, verbose=1,
          batch_size=128)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

#=============================== loss, accuracy ====================================#
# loss :  0.6028042435646057
# accuracy :  0.7942000031471252
#===================================================================================#

