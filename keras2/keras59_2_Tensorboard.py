import tensorboard
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from keras.layers import GlobalAveragePooling2D
import tensorflow as tf
from torch import Tensor
print(tf.__version__)


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.  
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.  

# from keras.utils.np_utils import to_categorical
# x_train = to_categorical(y_train)
# x_test =to_categorical(y_test)

#2. 모델
activation = 'relu'
drop = 0.2
optimizer = 'adam'

inputs = Input(shape=(28, 28, 1), name='input')
x = Conv2D(64, (2, 2), padding='valid', 
           activation=activation, name='hidden1')(inputs)   # 27, 27, 128
x = MaxPool2D()(x)                                         #  13, 13, 128
x = Dropout(drop)(x)
x = Conv2D(128, (2, 2), padding='same',                    #  27, 27, 64
           activation=activation, name='hidden2')(x)
x = MaxPool2D()(x)                                         #  13, 13, 128
x = Dropout(drop)(x)
x = Conv2D(256, (2, 2), padding='valid', 
           activation=activation, name='hidden3')(x)        # 12, 12, 256
x = MaxPool2D()(x)                                         #  13, 13, 128
x = Dropout(drop)(x)                                        
# x = Flatten()(x)                  # 25 * 25 * 32 = 20000
x = GlobalAveragePooling2D()(x)     # Flatten은 연산량이 많으므로 Pooling 커넥티드 레이어를 통해 연산량이 적으므로 성능이 좋을 수 있음

x = Dense(100, activation=activation, name='hidden4')(x)
x = Dropout(drop)(x)
x = Dense(64, activation=activation, name='hidden5')(x)
x = Dropout(drop)(x)
outputs = Dense(10, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)

#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard   # TensorBoard 임포트

es = EarlyStopping(monitor = 'val_loss', patience=15, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=7, mode='auto', verbose=1,
                              factor=0.5)
tb = TensorBoard(log_dir='d:/study_data/tensorboard_log/_graph', histogram_freq=0,  #histogram_freq=1 (기본적으로 꺼져 있음)으로 설정하면 모든 에포크마다 히스토그램 계산을 활성화
                 write_graph=True, write_images=True)
# TensorBoard 실행방법 : tensorboard --logdir=. (경로)
# http://localhost:6006
# http://127.0.0.:6006


from keras.optimizers import Adam
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
              metrics=['acc'])

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1,
                 callbacks=[es, reduce_lr, tb], validation_split=0.4)
end = time.time()

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)

print('learning_rate : ', learning_rate)
print('loss : ', round(loss, 4))
print('accuracy_score : ', round(acc, 4))
print('걸린시간 : ', end - start)

# learning_rate :  0.001
# loss :  0.0428
# accuracy_score :  0.9902
# 걸린시간 :  228.08930373191833

############################ 시각화 ###############################
import matplotlib.pyplot as plt

#1. 
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

#2. 
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])

plt.show()


