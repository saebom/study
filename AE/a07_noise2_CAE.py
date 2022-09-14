# [실습] 말하지 않아도 알아요!!!

import numpy as np
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D, UpSampling2D

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

################################ 노이즈 추가 #################################
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)  # 0~1 사이의 값으로 컷해줌
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)    # 0~1 사이의 값으로 컷해줌

##############################################################################

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
    model.add(UpSampling2D((2, 2), interpolation='nearest'))                                     # `"lanczos5"`, `"mitchellcubic"` 
    model.add(Conv2D(154, (2, 2), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(1, (4, 4), padding='same', activation='sigmoid'))
    model.summary()
    return model


model = autoencoder(hidden_layer_size=154)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train_noised, x_train, epochs=30)
output = model.predict(x_test_noised)


from matplotlib import pyplot as plt
import random 

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), 
      (ax11, ax12, ax13, ax14, ax15))  = \
    plt.subplots(3, 5, figsize=(20, 7))
    
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
  
# noise 추가된 이미지를 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("NOISE", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
          
# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
                
plt.tight_layout()
plt.show()    
