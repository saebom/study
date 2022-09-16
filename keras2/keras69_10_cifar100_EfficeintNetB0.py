# [실습] 전이학습, 안돌아가면 다른 모델 추가해서 돌려
# 02.Xception
# 03.ResNet50
# 04.ResNet101
# 05.InceptionV3
# 06.InceptionResNetV2
# 07.DenseNet121
# 08.MobileNetv2
# 09.NasNetMobile
# 10.EfficeintNetB0

from re import escape
import tensorflow as tf
import numpy as np
from keras.datasets import cifar100
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.layers import Input
from keras.callbacks import EarlyStopping
from keras.applications import EfficientNetB0
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()


# 2. 모델
model = Sequential()
input = Input(shape=(32, 32, 3))
efnb0 = EfficientNetB0(weights='imagenet', include_top=False)(input)
gap = GlobalAveragePooling2D()(efnb0)
hidden1 = Dense(128, activation='relu')(gap)
hidden2 = Dense(128, activation='relu')(hidden1)
output = Dense(100, activation='softmax')(hidden2)
model = Model(inputs=input, outputs=output)

# vgg16.trainable = False     
# model.trainable = False     

#3. 컴파일, 훈련
model.compile(optimizer='adam', metrics=['accuracy'], 
                loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                   restore_best_weights=True, verbose=1)
model.fit(x_train, y_train, epochs=200, validation_split=0.4, verbose=1,
          callbacks=[es], batch_size=128)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

# ====================================== #
# loss :  1.9655818939208984
# acc :  0.4975000023841858
# 기존 accuracy :  0.5112000107765198
# ====================================== #
