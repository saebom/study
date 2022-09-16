from re import X
from fsspec import Callback
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf
tf.random.set_seed(1234)

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# One Hot Encoding 
from tensorflow.python.keras.utils.np_utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)  # (150, 3)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=1234
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (105, 4) (45, 4) (105, 3) (45, 3)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(105, 2, 2, 1) 
x_test = x_test.reshape(45, 2, 2, 1)
print(x_train.shape)    
print(np.unique(x_train, return_counts=True))


#2. 모델
activation = 'relu'
drop = 0.25

inputs = Input(shape=(2, 2, 1), name='input')
x = Conv2D(64, (1, 1), padding='same', 
           activation=activation, name='hidden1')(inputs)   
x = MaxPool2D()(x)                                         
x = Dropout(drop)(x)
x = Conv2D(64, (1, 1), padding='same',                    
           activation=activation, name='hidden2')(x)
x = Dropout(drop)(x)
x = Conv2D(128, (1, 1), padding='same',                    
           activation=activation, name='hidden3')(x)
x = Dropout(drop)(x)
x = Conv2D(64, (1, 1), padding='same',                    
           activation=activation, name='hidden4')(x)
x = Dropout(drop)(x)
x = Conv2D(32, (1, 1), padding='same',                    
           activation=activation, name='hidden5')(x)
x = Dropout(drop)(x)                                       
x = GlobalAveragePooling2D()(x)     # Flatten은 연산량이 많으므로 Pooling 커넥티드 레이어를 통해 연산량이 적으므로 성능이 좋을 수 있음

x = Dense(128, activation=activation, name='hidden6')(x)
x = Dropout(drop)(x)
outputs = Dense(3, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Nadam
learning_rate = 0.01
optimizer = Adam(lr=learning_rate)

model.compile(optimizer=optimizer, metrics=['accuracy'],
                loss='categorical_crossentropy')

import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=500, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1,
                              factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=1000, validation_split=0.4, 
          callbacks=[es, reduce_lr], batch_size=32)
end = time.time()

#4. 평가, 예측
from sklearn.metrics import accuracy_score
loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
y_predict = y_predict.argmax(axis=1)
y_test = y_test.argmax(axis=1)

acc = accuracy_score(y_test, y_predict)

print('loss : ', loss)
print('acc : ', acc)
print('걸린시간 : ', end - start)

# =========================== 결과 ============================== #
# loss :  0.6784413456916809
# acc :  0.6222222222222222
# 걸린시간 :  29.10020637512207
# =============================================================== #