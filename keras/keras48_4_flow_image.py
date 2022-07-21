from click import argument
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
    fill_mode='nearest',
    featurewise_center=True,
    featurewise_std_normalization=True
)

# [실습]
# x_augmented 10개와 x_train 10개를 비교하는 이미지를 출력할 것!!!

# 증폭 사이즈
augment_size = 10
randidx = np.random.randint(x_train.shape[0], size=augment_size)

x_data = x_train[randidx].copy()
y_data = y_train[randidx].copy()
print(x_data.shape)    # (10, 28, 28)
print(y_data.shape)    # (10,)

x_train = x_data.reshape(10, 28, 28, 1)
x_augmented = x_data.reshape(x_data.shape[0], 
                             x_data.shape[1], 
                             x_data.shape[2], 1)

x_augmented = train_datagen.flow(x_augmented, y_data,
                                 batch_size=augment_size,
                                 shuffle=False).next()[0]   # next()[0] => x를 넣겠다는 의미
                                                            # shuffle=False이므로 label값 변환없이 들어감. 나중에 y값을 그대로 쓸 수 있음
print(x_augmented)
print(x_augmented.shape)    # (10, 28, 28, 1)
# 클래스는 괄호 2개, np.concatenate는 괄호 2개 사용해야 함


import matplotlib.pyplot as plt
plt.figure(figsize=(2,10))
for i in range(20):
    if i <= 9:
        plt.subplot(2, 10, i+1)
        plt.axis('off')
        plt.imshow(x_train[i], cmap='gray')
    else:
        plt.subplot(2, 10, i+1)
        plt.axis('off')
        plt.imshow(x_augmented[i-10], cmap='gray')
plt.show()

# import matplotlib.pyplot as plt
# plt.figure(figsize=(2,10))
# for i in range(20):
#     if i <= 9:
#         plt.subplot(2, 10, i+1)
#         plt.axis('off')
#         plt.imshow(x_train[i], cmap='gray')
#     else:
#         plt.subplot(2, 10, i+1)
#         plt.axis('off')
#         plt.imshow(x_train[i+60000-10], cmap='gray')
# plt.show()
