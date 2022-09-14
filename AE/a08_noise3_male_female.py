# [실습] keras47_4 남자 여자에 noise를 너서
# predict 첫번째 : 기미 주근깨 여드름 제거!!!
# random하게 5개 정도 원본/수정본 빼주고

# predict 첫번째 : 본인 사진너서 빼!!! // 원본 수정본

import numpy as np
from tensorflow.keras.preprocessing import image

#1. 데이터 로드
x_train = np.load('D:/study_data/_save/_npy/keras47_4_train_x.npy')
x_test = np.load('D:/study_data/_save/_npy/keras47_4_test_x.npy')
me_img = np.load('D:/study_data/_save/_npy/me_x.npy')


################################ 노이즈 추가 #################################
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
me_img_noised = me_img + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)  # 0~1 사이의 값으로 컷해줌
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)    # 0~1 사이의 값으로 컷해줌
me_img_noised = np.clip(me_img_noised, a_min=0, a_max=1)    # 0~1 사이의 값으로 컷해줌
##############################################################################

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2, 2),    
                    padding='same', 
                    activation='relu',
                    input_shape=(150, 150, 3)))      
    model.add(MaxPooling2D(2, 2))           
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))  
    model.add(UpSampling2D((2, 2), interpolation='nearest'))                                     # `"lanczos5"`, `"mitchellcubic"` 
    model.add(Conv2D(154, (2, 2), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(3, (4, 4), padding='same', activation='sigmoid'))
    model.summary()
    return model


model = autoencoder(hidden_layer_size=154)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train_noised, x_train, epochs=10)
output = model.predict(x_test_noised)

me_output = model.predict(me_img_noised)

from matplotlib import pyplot as plt
import random 

fig, ((ax1, ax2, ax3, ax4, ax5, me1), (ax6, ax7, ax8, ax9, ax10, me2), 
      (ax11, ax12, ax13, ax14, ax15, me3))  = \
    plt.subplots(3, 6, figsize=(20, 7))
    
# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(150, 150, 3), cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
me1.imshow(me_img[0])  

# noise 추가된 이미지를 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(150, 150, 3), cmap='gray')
    if i ==0:
        ax.set_ylabel("NOISE", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
me2.imshow(me_img_noised[0])   
       
# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(150, 150, 3), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
me3.imshow(me_output[0])                
plt.tight_layout()
plt.show()    

