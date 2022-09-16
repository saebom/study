from re import X
from fsspec import Callback
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf
print(tf.__version__)


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=72
)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (398, 30) (171, 30) (398,) (171,)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(398, 6, 5, 1) 
x_test = x_test.reshape(171, 6, 5, 1)
print(x_train.shape)    
print(np.unique(x_train, return_counts=True))

#2. 모델
activation = 'relu'
drop = 0.25
optimizer = 'adam'

inputs = Input(shape=(6,5,1), name='input')
x = Conv2D(64, (2, 2), padding='same', 
           activation=activation, name='hidden1')(inputs)   
x = MaxPool2D()(x)                                         
x = Dropout(drop)(x)
x = Conv2D(64, (3,3), padding='same',                    
           activation=activation, name='hidden2')(x)
x = MaxPool2D()(x)                                         
x = Dropout(drop)(x)
x = Conv2D(128, (3, 3), padding='same',                    
           activation=activation, name='hidden3')(x)
x = Dropout(drop)(x)
x = Conv2D(64, (3, 3), padding='same',                    
           activation=activation, name='hidden4')(x)
x = Dropout(drop)(x)
x = Conv2D(32, (3, 3), padding='same',                    
           activation=activation, name='hidden5')(x)
x = Dropout(drop)(x)                                       
# x = Flatten()(x)                  # 25 * 25 * 32 = 20000
x = GlobalAveragePooling2D()(x)     # Flatten은 연산량이 많으므로 Pooling 커넥티드 레이어를 통해 연산량이 적으므로 성능이 좋을 수 있음

x = Dense(128, activation=activation, name='hidden6')(x)
x = Dropout(drop)(x)
outputs = Dense(1, activation='linear', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()


#3. 컴파일, 훈련
model.compile(optimizer=optimizer, metrics=['accuracy'],
                loss='binary_crossentropy')

import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1,
                              factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=1000, validation_split=0.4, 
          callbacks=[es, reduce_lr], batch_size=128)
end = time.time()

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
print('걸린시간 : ', end - start)

# =========================== 결과 ============================== #
# loss :  0.4397708475589752
# acc :  0.9590643048286438
# 걸린시간 :  14.896881580352783
# =============================================================== #