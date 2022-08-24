# [실습] 요니 땅!!
# 2, 3, 7, 8, 9, 10, 11, 12

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
tf.compat.v1.set_random_seed(123)

#1. 데이터
datasets = load_diabetes()
x = datasets.data      # (569, 30)
y = datasets.target.reshape(-1, 1)    # (569,1)
print(x.shape, y.shape) # (1797, 64) (1797, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123)

print(type(x_train), type(y_train))
print(x_train.dtype, y_train.dtype) # float64 int32
print(x_test.dtype, y_test.dtype)   # float64 int32

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.zeros([30, 1]), name='weight') # weight는 random_normal이나 zeros를 사용해도 괜찮음
b = tf.compat.v1.Variable(tf.zeros([1]), name='bias')   # bias는 통상적으로 0 이므로 zeros를 사용
# w = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,1]), name='weight')
# b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)    # sigmoid 를 씌운다

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  # binary_crossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-6)   
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0005)   
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-7)   
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0000002)   
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session() 
sess.run(tf.compat.v1.global_variables_initializer())     

epochs = 2001
for step in range(epochs):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train], 
                                    feed_dict={x:x_train, y:y_train})
    if step %20 == 0:
        print(step, 'loss : ', cost_val, '\n', hy_val)      
            
#4. 평가, 예측
y_predict = sess.run(hypothesis, feed_dict={x:x_test})
print(y_predict)    # [4.00006676 5.00008345 6.00010014]

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)      # r2 :  

mae = mean_absolute_error(y_test, y_predict)
print('mae : ', mae)    # mae :  

sess.close()
    
