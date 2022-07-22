from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.datasets import cifar10
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
tf.random.set_seed(777)

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, 
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest',
    featurewise_center=True,
    featurewise_std_normalization=True, 
)

train_datagen2 = ImageDataGenerator(
    rescale=1./255  
)

# 증폭 사이즈
augment_size = 40000
batch_size = 90000    
randidx = np.random.randint(x_train.shape[0], size=augment_size)                                                                                               
print(x_train.shape) # (50000, 32, 32, 3)
print(x_train.shape[0]) # (50000)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)    # (40000, 32, 32, 3)
print(y_augmented.shape)    # (40000,)


x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
x_augmented = x_augmented.reshape(x_augmented.shape[0], 
                                  x_augmented.shape[1], 
                                  x_augmented.shape[2], 3)

x_augmented = train_datagen.flow(x_augmented, y_augmented,
                              batch_size=augment_size,
                              shuffle=False).next()[0]      # next()[0] => x를 넣겠다는 의미
                                                            # shuffle=False이므로 label값 변환없이 들어감. 나중에 y값을 그대로 쓸 수 있음
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

xy_train = train_datagen2.flow(x_train, y_train,
                              batch_size=batch_size,
                              shuffle=False)

xy_test = train_datagen2.flow(x_test, y_test,
                              batch_size=batch_size,
                              shuffle=False)


np.save('d:/study_data/_save/_npy/keras49_3_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/keras49_3_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/keras49_3_test_x.npy', arr=xy_test[0][0])
np.save('d:/study_data/_save/_npy/keras49_3_test_y.npy', arr=xy_test[0][1])

'''
#2. 모델링 
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), # padding='same', 
                 activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(2, 2))  
model.add(Dropout(0.2))     
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))                
model.add(MaxPooling2D(2, 2))  
model.add(Dropout(0.2))     
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(2, 2))     
model.add(Dropout(0.2))     
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))   
model.add(MaxPooling2D(2, 2))     
model.add(Dropout(0.2))                    
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))  
# model.add(Dropout(0.2))   
             
model.add(Flatten())    
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='Nadam', 
              metrics=['accuracy'])

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
print(date)

filepath = './_ModelCheckPoint/k28/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '02_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=128,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

y_predict = model.predict(x_test)
y_predict = y_predict.argmax(axis=1)      
y_test = y_test.argmax(axis=1)            

acc = accuracy_score(y_test, y_predict)
print("=====================================================================")   
print('acc 스코어 : ', acc)  

print("=====================================================================")
print("걸린시간 : ", end_time)


#그래프로 비교
font_path = 'C:\Windows\Fonts\malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
# plt.title('loss & val_loss')    
plt.title('로스값과 검증로스값')    
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()   
plt.show()


#=============================== loss, accuracy ====================================#
# loss :  0.6028042435646057
# accuracy :  0.7942000031471252
#===================================================================================#
'''