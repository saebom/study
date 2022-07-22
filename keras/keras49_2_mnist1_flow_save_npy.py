from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, 
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

train_datagen2 = ImageDataGenerator(
    rescale=1./255  
)

# 증폭 사이즈
augment_size = 40000
batch_size = 100000     # save 파일 저장 시 전체 이미지 갯수로 batch_size 맞춰줌
randidx = np.random.randint(x_train.shape[0], size=augment_size)    # (60000, 40000)
                                                                    # np.random.randint는 랜덤하게 int를 뽑아냄
                                                                    # x_train.shape[0] = 60000
                                                                    # x_train.shape[1] = 28
                                                                    # x_train.shape[2] = 28                                
print(x_train.shape) # (60000, 28, 28)
print(x_train.shape[0]) # (60000)
print(randidx)          # [31720 43951 44299 ... 22547 15575 47042]
print(np.min(randidx), np.max(randidx)) # 3 59999
print(type(randidx))    # <class 'numpy.ndarray'>

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)    # (40000, 28, 28)
print(y_augmented.shape)    # (40000,)


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_augmented = x_augmented.reshape(x_augmented.shape[0], 
                                  x_augmented.shape[1], 
                                  x_augmented.shape[2], 1)

x_augmented = train_datagen.flow(x_augmented, y_augmented,
                              batch_size=augment_size,
                              shuffle=False).next()[0]      # next()[0] => x를 넣겠다는 의미
                                                            # shuffle=False이므로 label값 변환없이 들어감. 나중에 y값을 그대로 쓸 수 있음
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)

xy_train = train_datagen2.flow(x_train, y_train,
                              batch_size=batch_size,
                              shuffle=False)

xy_test = train_datagen2.flow(x_test, y_test,
                              batch_size=batch_size,
                              shuffle=False)


np.save('d:/study_data/_save/_npy/keras49_2_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/keras49_2_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/keras49_2_test_x.npy', arr=xy_test[0][0])
np.save('d:/study_data/_save/_npy/keras49_2_test_y.npy', arr=xy_test[0][1])

'''
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
model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
print(date)

filepath = './_ModelCheckPoint/k28/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '01_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=50, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

import tensorflow as tf
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)    
y_test = tf.argmax(y_test, axis=1)          



#=============================== loss, accuracy ====================================#
# loss :  0.04272077605128288
# accuracy :  0.9915000200271606
#===================================================================================#
'''