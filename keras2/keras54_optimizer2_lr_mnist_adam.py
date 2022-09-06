# 맹그러봐
# optimizer, learning_rate 갱신!!!


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.datasets import mnist
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical

import tensorflow as tf
import time


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.  
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.  

# One Hot Encoding
import pandas as pd
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

#2. 모델링 
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4, 4),    
                 padding='same', 
                 input_shape=(28, 28, 1)))      
model.add(MaxPooling2D(2, 2))           
model.add(Dropout(0.2))
model.add(Conv2D(64, (4, 4), padding='valid', activation='relu'))                
model.add(MaxPooling2D(2, 2))          
model.add(Dropout(0.2))
model.add(Conv2D(64, (4, 4), padding='same', activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())    # (N, 63)  (N, 175)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))



#3. 컴파일, 훈련
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax
from tensorflow.python.keras.optimizer_v2 import rmsprop, nadam

learning_rates = [0.001]
for lr in learning_rates:
    optimizer1 = adam.Adam(learning_rate=lr)            
    optimizer2 = adadelta.Adadelta(learning_rate=lr)    
    optimizer3 = adagrad.Adagrad(learning_rate=lr)      
    optimizer4 = adamax.Adamax(learning_rate=lr)
    optimizer5 = rmsprop.RMSProp(learning_rate=lr)
    optimizer6 = nadam.Nadam(learning_rate=lr)

    optimizers = [optimizer1,optimizer2, optimizer3, optimizer4, optimizer5, optimizer6]

    for optimizer in optimizers:

        model.compile(loss = 'categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])

        import datetime
        date = datetime.datetime.now()      
        date = date.strftime("%m%d_%H%M")  
        print(date)

        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min',
                                      restore_best_weights=True,
                                      verbose=0)

        start_time = time.time()
        hist = model.fit(x_train, y_train, epochs=30, batch_size=64,
                         validation_split=0.2,
                         callbacks=[earlyStopping],
                         verbose=0)
        end_time = time.time() - start_time


        #4. 평가, 예측
        import numpy as np
        loss, acc = model.evaluate(x_test,y_test)
        y_predict = model.predict(x_test)
        y_predict = np.argmax(y_predict,axis=1)
        y_predict = to_categorical(y_predict)
        
        optimizer_name = optimizer.__class__.__name__
        # acc = accuracy_score(y_test, y_predict)
        print('{0} lr : {1}'.format(optimizer_name, lr),
              'acc : ', round(acc, 4), 
              'loss : ', round(loss, 4),
              '걸린 시간 : ', round(end_time, 4))  

# ============================== 결과 ================================= #
# Adam lr : 0.1 acc :  0.0982 loss :  2.3046 걸린 시간 :  100.0994
# Adadelta lr : 0.1 acc :  0.1135 loss :  2.3011 걸린 시간 :  64.8618
# Adagrad lr : 0.1 acc :  0.1135 loss :  2.3011 걸린 시간 :  64.4475
# Adamax lr : 0.1 acc :  0.1028 loss :  2.3022 걸린 시간 :  39.8358
# RMSprop lr : 0.1 acc :  0.1032 loss :  2.3035 걸린 시간 :  77.6424
# Nadam lr : 0.1 acc :  0.1028 loss :  2.304 걸린 시간 :  132.7813

# Adam lr : 0.01 acc :  0.1028 loss :  2.3018 걸린 시간 :  102.3022
# Adadelta lr : 0.01 acc :  0.1135 loss :  2.3014 걸린 시간 :  99.7314
# 0906_0029
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3011 - accuracy: 0.1135
# Adagrad lr : 0.01 acc :  0.1135 loss :  2.3011 걸린 시간 :  37.7809
# 0906_0030
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3011 - accuracy: 0.1135
# Adamax lr : 0.01 acc :  0.1135 loss :  2.3011 걸린 시간 :  45.983
# 0906_0030
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3017 - accuracy: 0.1028
# RMSprop lr : 0.01 acc :  0.1028 loss :  2.3017 걸린 시간 :  89.3748
# 0906_0032
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3016 - accuracy: 0.1135
# Nadam lr : 0.01 acc :  0.1135 loss :  2.3016 걸린 시간 :  161.3867
# 0906_0035
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adam lr : 0.001 acc :  0.1135 loss :  2.301 걸린 시간 :  39.4599
# 0906_0035
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adadelta lr : 0.001 acc :  0.1135 loss :  2.301 걸린 시간 :  38.2587
# 0906_0036
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adagrad lr : 0.001 acc :  0.1135 loss :  2.301 걸린 시간 :  37.978
# 0906_0037
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adamax lr : 0.001 acc :  0.1135 loss :  2.301 걸린 시간 :  80.2779
# 0906_0038
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# RMSprop lr : 0.001 acc :  0.1135 loss :  2.301 걸린 시간 :  151.4909
# 0906_0041
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# Nadam lr : 0.001 acc :  0.1135 loss :  2.301 걸린 시간 :  223.7375
# 0906_0044
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adam lr : 0.0001 acc :  0.1135 loss :  2.301 걸린 시간 :  39.2516
# 0906_0045
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# Adadelta lr : 0.0001 acc :  0.1135 loss :  2.301 걸린 시간 :  38.9126
# 0906_0046
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adagrad lr : 0.0001 acc :  0.1135 loss :  2.301 걸린 시간 :  41.2998
# 0906_0046
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adamax lr : 0.0001 acc :  0.1135 loss :  2.301 걸린 시간 :  39.2691
# 0906_0047
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# RMSprop lr : 0.0001 acc :  0.1135 loss :  2.301 걸린 시간 :  64.8297
# 0906_0048
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# Nadam lr : 0.0001 acc :  0.1135 loss :  2.301 걸린 시간 :  96.4138
# 0906_0050
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adam lr : 1e-05 acc :  0.1135 loss :  2.301 걸린 시간 :  39.3838
# 0906_0051
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adadelta lr : 1e-05 acc :  0.1135 loss :  2.301 걸린 시간 :  38.56
# 0906_0051
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adagrad lr : 1e-05 acc :  0.1135 loss :  2.301 걸린 시간 :  58.4681
# 0906_0052
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adamax lr : 1e-05 acc :  0.1135 loss :  2.301 걸린 시간 :  46.254
# 0906_0053
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# RMSprop lr : 1e-05 acc :  0.1135 loss :  2.301 걸린 시간 :  70.5327
# 0906_0054
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# Nadam lr : 1e-05 acc :  0.1135 loss :  2.301 걸린 시간 :  96.4241
# 0906_0056
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3150 - accuracy: 0.1009
# Adam lr : 0.5 acc :  0.1009 loss :  2.315 걸린 시간 :  57.3965
# 0906_0057
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adadelta lr : 0.5 acc :  0.1135 loss :  2.301 걸린 시간 :  57.9186
# 0906_0058
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3012 - accuracy: 0.1135
# Adagrad lr : 0.5 acc :  0.1135 loss :  2.3012 걸린 시간 :  81.7729
# 0906_0059
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3082 - accuracy: 0.0982
# Adamax lr : 0.5 acc :  0.0982 loss :  2.3082 걸린 시간 :  70.4261
# 0906_0100
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3138 - accuracy: 0.1135
# RMSprop lr : 0.5 acc :  0.1135 loss :  2.3138 걸린 시간 :  134.081
# 0906_0103
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3213 - accuracy: 0.1010
# Nadam lr : 0.5 acc :  0.101 loss :  2.3213 걸린 시간 :  260.8696
# 0906_0107
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3035 - accuracy: 0.1010
# Adam lr : 0.05 acc :  0.101 loss :  2.3035 걸린 시간 :  43.0506
# 0906_0108
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adadelta lr : 0.05 acc :  0.1135 loss :  2.301 걸린 시간 :  96.5474
# 0906_0109
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adagrad lr : 0.05 acc :  0.1135 loss :  2.301 걸린 시간 :  51.8657
# 0906_0110
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3034 - accuracy: 0.1028
# Adamax lr : 0.05 acc :  0.1028 loss :  2.3034 걸린 시간 :  49.3517
# 0906_0111
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3040 - accuracy: 0.1135
# RMSprop lr : 0.05 acc :  0.1135 loss :  2.304 걸린 시간 :  111.756
# 0906_0113
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3019 - accuracy: 0.1135
# Nadam lr : 0.05 acc :  0.1135 loss :  2.3019 걸린 시간 :  96.3995
# 0906_0115
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adam lr : 0.005 acc :  0.1135 loss :  2.301 걸린 시간 :  85.5929
# 0906_0116
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adadelta lr : 0.005 acc :  0.1135 loss :  2.301 걸린 시간 :  103.0152
# 0906_0118
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3010 - accuracy: 0.1135
# Adagrad lr : 0.005 acc :  0.1135 loss :  2.301 걸린 시간 :  37.8343
# 0906_0119
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3012 - accuracy: 0.1135
# Adamax lr : 0.005 acc :  0.1135 loss :  2.3012 걸린 시간 :  55.7533
# 0906_0120
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3011 - accuracy: 0.1135
# RMSprop lr : 0.005 acc :  0.1135 loss :  2.3011 걸린 시간 :  172.1852
# 0906_0122
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3016 - accuracy: 0.1028
# Nadam lr : 0.005 acc :  0.1028 loss :  2.3016 걸린 시간 :  234.6761
# 0906_0126
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3011 - accuracy: 0.1135
# Adam lr : 0.0005 acc :  0.1135 loss :  2.3011 걸린 시간 :  39.5376
# 0906_0127
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3011 - accuracy: 0.1135
# Adadelta lr : 0.0005 acc :  0.1135 loss :  2.3011 걸린 시간 :  41.3309
# 0906_0128
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3011 - accuracy: 0.1135
# Adagrad lr : 0.0005 acc :  0.1135 loss :  2.3011 걸린 시간 :  37.3224
# 0906_0128
# 313/313 [==============================] - 1s 3ms/step - loss: 2.3011 - accuracy: 0.1135
# Adamax lr : 0.0005 acc :  0.1135 loss :  2.3011 걸린 시간 :  38.0167
# 0906_0129
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# RMSprop lr : 0.0005 acc :  0.1135 loss :  2.301 걸린 시간 :  64.3987
# 0906_0130
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# Nadam lr : 0.0005 acc :  0.1135 loss :  2.301 걸린 시간 :  177.1358




