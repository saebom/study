from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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

# 증폭 사이즈
augment_size = 40
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

import time
start_time = time.time()
print("시작!!!")
x_augmented = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size=augment_size,
                                 save_to_dir='d:/study_data/_temp/',    # 증폭이미지 저장
                                 shuffle=False).next()[0]   
                                                            
end_time = time.time() - start_time
print(augment_size, " 증폭에 걸린 시간 : ", round(end_time, 3), "초")

x_train = np.concatenate((x_train, x_augmented))    # 클래스는 괄호 2개, np.concatenate는 괄호 2개 사용해야 함
y_train = np.concatenate((y_train, y_augmented))
 
print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)

