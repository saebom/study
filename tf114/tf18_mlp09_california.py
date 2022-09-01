import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
tf.compat.v1.set_random_seed(777)


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data      
y = datasets.target.reshape(-1, 1)    
print(x.shape, y.shape) # (506, 13) (506, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=134)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(type(x_train), type(y_train))
print(x_train.dtype, y_train.dtype) # float64 int32
print(x_test.dtype, y_test.dtype)   # float64 int32

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random_normal([8, 32]), name='weight1') 
b1 = tf.compat.v1.Variable(tf.random_normal([32]), name='bias1')   

hidden_layer1 = tf.compat.v1.matmul(x, w1) + b1

w2 = tf.compat.v1.Variable(tf.random_normal([32, 64]), name='weight2') 
b2 = tf.compat.v1.Variable(tf.random_normal([64]), name='bias2')   

hidden_layer2 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden_layer1, w2) + b2)

w3 = tf.compat.v1.Variable(tf.random_normal([64, 128]), name='weight3') 
b3 = tf.compat.v1.Variable(tf.random_normal([128]), name='bias3')   

hidden_layer3 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden_layer2, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random_normal([128, 128]), name='weight4') 
b4 = tf.compat.v1.Variable(tf.random_normal([128]), name='bias4')   

hidden_layer3 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden_layer3, w4) + b4)

# output layer
w5 = tf.compat.v1.Variable(tf.compat.v1.zeros([128,1]), name='weight5')
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias5')

hypothesis = tf.compat.v1.matmul(hidden_layer3, w5) + b5

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)   
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)   
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
print('r2 : ', r2)      # r2 :  0.6661088487503457

mae = mean_absolute_error(y_test, y_predict)
print('mae : ', mae)    # 0.44594007144042397

sess.close()
    
