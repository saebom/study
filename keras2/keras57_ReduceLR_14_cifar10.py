from re import X
from fsspec import Callback
import numpy as np
from keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf
print(tf.__version__)
import tensorflow as tf
tf.random.set_seed(777)

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000, 32*32*3) 
x_test = x_test.reshape(10000, 32*32*3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3) 
x_test = x_test.reshape(10000, 32, 32, 3)

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test =to_categorical(y_test)

#2. 모델
activation = 'relu'
drop = 0.2
optimizer = 'adam'

inputs = Input(shape=(32, 32, 3), name='input')
x = Conv2D(32, (2, 2), padding='valid', 
           activation=activation, name='hidden1')(inputs)   # 27, 27, 128
x = MaxPool2D()(x)                                         #  13, 13, 128
x = Dropout(drop)(x)
x = Conv2D(32, (2, 2), padding='same',                    #  27, 27, 64
           activation=activation, name='hidden2')(x)
x = MaxPool2D()(x)                                         #  13, 13, 128
x = Dropout(drop)(x)
x = Conv2D(64, (2, 2), padding='same',                    #  27, 27, 64
           activation=activation, name='hidden3')(x)
x = MaxPool2D()(x)                                         #  13, 13, 128
x = Dropout(drop)(x)
x = Conv2D(64, (2, 2), padding='same',                    #  27, 27, 64
           activation=activation, name='hidden4')(x)
x = MaxPool2D()(x)                                         #  13, 13, 128
x = Dropout(drop)(x)
x = Conv2D(128, (2, 2), padding='same',                    #  27, 27, 64
           activation=activation, name='hidden5')(x)
                                       
# x = Flatten()(x)                  # 25 * 25 * 32 = 20000
x = GlobalAveragePooling2D()(x)     # Flatten은 연산량이 많으므로 Pooling 커넥티드 레이어를 통해 연산량이 적으므로 성능이 좋을 수 있음

x = Dense(128, activation=activation, name='hidden6')(x)
x = Dropout(drop)(x)
outputs = Dense(10, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()


#3. 컴파일, 훈련
model.compile(optimizer=optimizer, metrics=['accuracy'],
                loss='categorical_crossentropy')

import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1,
                              factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=300, validation_split=0.4, 
          callbacks=[es, reduce_lr], batch_size=128)
end = time.time()

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
print('걸린시간 : ', end - start)


# loss :  0.8352065682411194
# acc :  0.7053999900817871
# 걸린시간 :  432.94375228881836
# Epoch 00162: ReduceLROnPlateau reducing learning rate to 3.125000148429535e-05.