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
x = datasets.data      
y = datasets.target.reshape(-1, 1)    
print(x.shape, y.shape) # (442, 10) (442, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123)

print(type(x_train), type(y_train))
print(x_train.dtype, y_train.dtype) # float64 int32
print(x_test.dtype, y_test.dtype)   # float64 int32

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')
hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.7)   
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
print(y_predict)   

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)      # r2 :  0.5650995686049121

mae = mean_absolute_error(y_test, y_predict)
print('mae : ', mae)    # mae :  42.928504086612314

sess.close()
    
