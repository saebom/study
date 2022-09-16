# [실습] 4번 카피 복붙
# CNN으로 딥하게 구성
# UpSampling 찾아서 이해하고 반드시 추가할 것 !!!
# UpSampling => padding 채우기(x), Nearest Neighbor(같은 값으로 채우기), 
#               Bilinear(선형 보간법으로 채우기), gaussian(정규분포값으로 채우기)

import numpy as np
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D, UpSampling2D

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

from keras.models import Sequential, Model
from keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2, 2),    
                    padding='same', 
                    activation='relu',
                    input_shape=(28, 28, 1)))      
    model.add(MaxPooling2D(2, 2))           
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))  
    model.add(UpSampling2D((2, 2), interpolation='gaussian'))  # UpSampling => interpolation의 종류 : `"nearest(디폴트)", "area"`, `"bicubic"`, 
                                                                                                    # `"bilinear"`,`"gaussian"`, `"lanczos3"`, 
                                                                                                    # `"lanczos5"`, `"mitchellcubic"` 
    model.add(Conv2D(154, (2, 2), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(1, (4, 4), padding='same', activation='sigmoid'))
    model.summary()
    return model


model = autoencoder(hidden_layer_size=64)
# model = autoencoder(hidden_layer_size=154)  # PCA의 95% 성능 154개, PCA를 통해 차원축소시 성능 확인하고 autoencoder
# model = autoencoder(hidden_layer_size=331)  # PCA의 99% 성능 331개

model.compile(optimizer='adam', loss='binary_crossentropy')
# model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=30)
output = model.predict(x_test)


from matplotlib import pyplot as plt
import random 

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10))  = \
    plt.subplots(2, 5, figsize=(20, 7))
    
# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
  
# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
plt.tight_layout()
plt.show()    
