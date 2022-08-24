# [실습] 요니 땅!!
# 2, 3, 7, 8, 9, 10, 11, 12

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
tf.compat.v1.set_random_seed(123)

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data      # (569, 30)
y = datasets.target    # (569,1)
y = y.astype(np.float)
y = y.reshape(-1, 1)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, stratify=y
)

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
# loss = tf.reduce_mean(tf.square(hypothesis - y))  # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  # binary_crossentropy
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
y_predict = sess.run(tf.cast(hy_val>0.5, dtype=tf.float32))   # tf.cast는 텐서를 새로운 형태로 캐스팅하는데 사용
                                                              # Boolean형태인 경우 True이면 1, False이면 0을 출력
from sklearn.metrics import accuracy_score, mean_absolute_error
acc = accuracy_score(y_train, y_predict)
print('acc : ', acc)      # acc :  0.9121265377855887
mae = mean_absolute_error(y_train, hy_val)
print('mae : ', mae)      # mae :  0.4426241162476933
sess.close()

