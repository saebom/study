# [실습]
# DNN으로 구성!!!

import tensorflow as tf
import numpy as np
import pandas as pd
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
tf.compat.v1.set_random_seed(123)


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.  # 1~0사이로 수렴
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.  # 1~0사이로 수렴

print(type(x_train), type(y_train))
print(x_train.dtype, y_train.dtype) # float64 int32
print(x_test.dtype, y_test.dtype)   # float64 int32

#2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 28*28]) 
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 10])

w1 = tf.compat.v1.Variable(tf.random_normal([784, 32]), name='weight1') 
b1 = tf.compat.v1.Variable(tf.random_normal([32]), name='bias1')   

hidden_layer1 = tf.compat.v1.matmul(x, w1) + b1

w2 = tf.compat.v1.Variable(tf.random_normal([32, 64]), name='weight2') 
b2 = tf.compat.v1.Variable(tf.random_normal([64]), name='bias2')   

hidden_layer2 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden_layer1, w2) + b2)

w3 = tf.compat.v1.Variable(tf.random_normal([64, 128]), name='weight3') 
b3 = tf.compat.v1.Variable(tf.random_normal([128]), name='bias3')   

hidden_layer3 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden_layer2, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random_normal([128, 64]), name='weight4') 
b4 = tf.compat.v1.Variable(tf.random_normal([64]), name='bias4')   

hidden_layer4 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden_layer3, w4) + b4)

# output layer
w5 = tf.compat.v1.Variable(tf.compat.v1.zeros([64,10]), name='weight5')
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name='bias5')

hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(hidden_layer4, w5) + b5)

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)   
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session() 
sess.run(tf.compat.v1.global_variables_initializer())     

epochs = 1001
for step in range(epochs):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train], 
                                    feed_dict={x:x_train, y:y_train})
    if step %20 == 0:
        print(step, 'loss : ', cost_val, '\n', hy_val)      
            
#4. 평가, 예측
y_predict = sess.run(tf.math.argmax(sess.run(hypothesis, feed_dict={x:x_test}), axis=1)) 
y_test = sess.run(tf.math.argmax(y_test, axis=1))

from sklearn.metrics import accuracy_score, mean_absolute_error
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)      # acc :  0.9541

mae = mean_absolute_error(y_test, y_predict)
print('mae : ', mae)    # mae :  0.1704

sess.close()
    
