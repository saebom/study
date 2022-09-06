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

learning_rates = [0.001, 0.0001, 0.00001, 0.1, 0.01, 0.5, 0.05, 0.005, 0.0005]
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


# Adam lr : 0.001 acc :  0.9926 loss :  0.0259 걸린 시간 :  77.7032
# 0906_0929
# 313/313 [==============================] - 1s 4ms/step - loss: 0.0222 - accuracy: 0.9937
# Adadelta lr : 0.001 acc :  0.9937 loss :  0.0222 걸린 시간 :  112.6777
# 0906_0931
# 313/313 [==============================] - 1s 4ms/step - loss: 0.0198 - accuracy: 0.9940
# Adagrad lr : 0.001 acc :  0.994 loss :  0.0198 걸린 시간 :  94.0413
# 0906_0932
# 313/313 [==============================] - 1s 4ms/step - loss: 0.0206 - accuracy: 0.9947
# Adamax lr : 0.001 acc :  0.9947 loss :  0.0206 걸린 시간 :  63.8617
# 0906_0933
# 313/313 [==============================] - 1s 4ms/step - loss: 0.0264 - accuracy: 0.9937
# RMSprop lr : 0.001 acc :  0.9937 loss :  0.0264 걸린 시간 :  81.1979
# 0906_0935
# 313/313 [==============================] - 1s 3ms/step - loss: 0.0262 - accuracy: 0.9919
# Nadam lr : 0.001 acc :  0.9919 loss :  0.0262 걸린 시간 :  149.8688
# 0906_0937
# 313/313 [==============================] - 1s 4ms/step - loss: 0.0204 - accuracy: 0.9943
# Adam lr : 0.0001 acc :  0.9943 loss :  0.0204 걸린 시간 :  46.8363
# 0906_0938
# 313/313 [==============================] - 1s 4ms/step - loss: 0.0204 - accuracy: 0.9943
# Adadelta lr : 0.0001 acc :  0.9943 loss :  0.0204 걸린 시간 :  42.6238
# 0906_0939
# 313/313 [==============================] - 1s 4ms/step - loss: 0.0203 - accuracy: 0.9944
# Adagrad lr : 0.0001 acc :  0.9944 loss :  0.0203 걸린 시간 :  91.4271
# 0906_0940
# 313/313 [==============================] - 1s 4ms/step - loss: 0.0204 - accuracy: 0.9946
# Adamax lr : 0.0001 acc :  0.9946 loss :  0.0204 걸린 시간 :  51.6371
# 0906_0941
# 313/313 [==============================] - 1s 4ms/step - loss: 0.0231 - accuracy: 0.9950
# RMSprop lr : 0.0001 acc :  0.995 loss :  0.0231 걸린 시간 :  66.8022
# 0906_0942
# 313/313 [==============================] - 1s 4ms/step - loss: 0.0200 - accuracy: 0.9947
# Nadam lr : 0.0001 acc :  0.9947 loss :  0.02 걸린 시간 :  184.2368
# 0906_0946
# 313/313 [==============================] - 1s 4ms/step - loss: 0.0206 - accuracy: 0.9954
# Adam lr : 1e-05 acc :  0.9954 loss :  0.0206 걸린 시간 :  41.663
# 0906_0946
# 313/313 [==============================] - 1s 4ms/step - loss: 0.0206 - accuracy: 0.9954
# Adadelta lr : 1e-05 acc :  0.9954 loss :  0.0206 걸린 시간 :  39.0763
# 0906_0947
# 313/313 [==============================] - 1s 4ms/step - loss: 0.0206 - accuracy: 0.9954
# Adagrad lr : 1e-05 acc :  0.9954 loss :  0.0206 걸린 시간 :  40.9284
# 0906_0948
# 313/313 [==============================] - 1s 4ms/step - loss: 0.0207 - accuracy: 0.9952
# Adamax lr : 1e-05 acc :  0.9952 loss :  0.0207 걸린 시간 :  44.5709
# 0906_0948
# 313/313 [==============================] - 1s 4ms/step - loss: 0.0214 - accuracy: 0.9951
# RMSprop lr : 1e-05 acc :  0.9951 loss :  0.0214 걸린 시간 :  68.5576
# 0906_0950
# 313/313 [==============================] - 1s 4ms/step - loss: 0.0216 - accuracy: 0.9953
# Nadam lr : 1e-05 acc :  0.9953 loss :  0.0216 걸린 시간 :  112.4919
# 0906_0952
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3036 - accuracy: 0.1028
# Adam lr : 0.1 acc :  0.1028 loss :  2.3036 걸린 시간 :  52.3214
# 0906_0952
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3014 - accuracy: 0.1135
# Adadelta lr : 0.1 acc :  0.1135 loss :  2.3014 걸린 시간 :  55.4619
# 0906_0953
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# Adagrad lr : 0.1 acc :  0.1135 loss :  2.301 걸린 시간 :  60.548
# 0906_0954
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3014 - accuracy: 0.1135
# Adamax lr : 0.1 acc :  0.1135 loss :  2.3014 걸린 시간 :  42.5254
# 0906_0955
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3071 - accuracy: 0.1028
# RMSprop lr : 0.1 acc :  0.1028 loss :  2.3071 걸린 시간 :  87.238
# 0906_0957
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3028 - accuracy: 0.1135
# Nadam lr : 0.1 acc :  0.1135 loss :  2.3028 걸린 시간 :  103.4092
# 0906_0958
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3015 - accuracy: 0.1135
# Adam lr : 0.01 acc :  0.1135 loss :  2.3015 걸린 시간 :  83.9639
# 0906_1000
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3011 - accuracy: 0.1135
# Adadelta lr : 0.01 acc :  0.1135 loss :  2.3011 걸린 시간 :  118.0286
# 0906_1002
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# Adagrad lr : 0.01 acc :  0.1135 loss :  2.301 걸린 시간 :  43.4553
# 0906_1003
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3013 - accuracy: 0.1135
# Adamax lr : 0.01 acc :  0.1135 loss :  2.3013 걸린 시간 :  89.2003
# 0906_1004
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3019 - accuracy: 0.1135
# RMSprop lr : 0.01 acc :  0.1135 loss :  2.3019 걸린 시간 :  69.2282
# 0906_1005
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3014 - accuracy: 0.1135
# Nadam lr : 0.01 acc :  0.1135 loss :  2.3014 걸린 시간 :  248.981
# 0906_1010
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3180 - accuracy: 0.0974
# Adam lr : 0.5 acc :  0.0974 loss :  2.318 걸린 시간 :  117.9524
# 0906_1012
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# Adadelta lr : 0.5 acc :  0.1135 loss :  2.301 걸린 시간 :  66.6701
# 0906_1013
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3011 - accuracy: 0.1135
# Adagrad lr : 0.5 acc :  0.1135 loss :  2.3011 걸린 시간 :  116.3609
# 0906_1015
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3083 - accuracy: 0.0982
# Adamax lr : 0.5 acc :  0.0982 loss :  2.3083 걸린 시간 :  77.9226
# 0906_1016
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3213 - accuracy: 0.1028
# RMSprop lr : 0.5 acc :  0.1028 loss :  2.3213 걸린 시간 :  78.789
# 0906_1017
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3099 - accuracy: 0.0980
# Nadam lr : 0.5 acc :  0.098 loss :  2.3099 걸린 시간 :  135.6871
# 0906_1020
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3042 - accuracy: 0.1028
# Adam lr : 0.05 acc :  0.1028 loss :  2.3042 걸린 시간 :  92.7982
# 0906_1021
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# Adadelta lr : 0.05 acc :  0.1135 loss :  2.301 걸린 시간 :  98.9819
# 0906_1023
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3011 - accuracy: 0.1135
# Adagrad lr : 0.05 acc :  0.1135 loss :  2.3011 걸린 시간 :  44.4848
# 0906_1024
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3022 - accuracy: 0.0958
# Adamax lr : 0.05 acc :  0.0958 loss :  2.3022 걸린 시간 :  88.4744
# 0906_1025
# 313/313 [==============================] - 2s 5ms/step - loss: 2.3023 - accuracy: 0.1010
# RMSprop lr : 0.05 acc :  0.101 loss :  2.3023 걸린 시간 :  113.5278
# 0906_1027
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3026 - accuracy: 0.1135
# Nadam lr : 0.05 acc :  0.1135 loss :  2.3026 걸린 시간 :  113.6738
# 0906_1029
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3011 - accuracy: 0.1135
# Adam lr : 0.005 acc :  0.1135 loss :  2.3011 걸린 시간 :  114.4958
# 0906_1031
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# Adadelta lr : 0.005 acc :  0.1135 loss :  2.301 걸린 시간 :  116.0693
# 0906_1033
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# Adagrad lr : 0.005 acc :  0.1135 loss :  2.301 걸린 시간 :  43.7303
# 0906_1034
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# Adamax lr : 0.005 acc :  0.1135 loss :  2.301 걸린 시간 :  51.6994
# 0906_1035
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3011 - accuracy: 0.1135
# RMSprop lr : 0.005 acc :  0.1135 loss :  2.3011 걸린 시간 :  149.2727
# 0906_1037
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3013 - accuracy: 0.1135
# Nadam lr : 0.005 acc :  0.1135 loss :  2.3013 걸린 시간 :  120.9461
# 0906_1039
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3010 - accuracy: 0.1135
# Adam lr : 0.0005 acc :  0.1135 loss :  2.301 걸린 시간 :  43.339
# 0906_1040
# 313/313 [==============================] - 1s 4ms/step - loss: 2.3011 - accuracy: 0.1135
# Adadelta lr : 0.0005 acc :  0.1135 loss :  2.3011 걸린 시간 :  99.1534




