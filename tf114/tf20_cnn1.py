import tensorflow as tf
import numpy as np

tf.compat.v1.set_random_seed(123)

#1. 데이터
# pip install keras==2.3
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.  # 1~0사이로 수렴
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.  # 1~0사이로 수렴

#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1]) # input_shape
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 64])  # tf.get_variable()함수는 tf.Variable을 직접호출 대신 변수를 가져오거나 생성하는 데 사용함
                                          # 2, 2 는 kernal 사이즈, 1은 color(chanel), 64는 output(filter)
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID')   
                                # 두번째, 세번째가 실제 stride 양쪽은 shape을 맞춰주는 것임
# model.add(Conv2d(64, kernel_size=(2,2), input_shape=(28,28,1))) 과 동일

print(w1)   # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1)   # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)