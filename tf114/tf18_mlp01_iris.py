#[실습] 12개 만들어 보아요

import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target.reshape(-1, 1)
print(x.shape, y.shape)  # (150, 4), (150, 1)

ohe = OneHotEncoder()
ohe.fit(y)
y = ohe.transform(y).toarray()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66, shuffle=True
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([4, 32]), name='weight1') 
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32]), name='bias1')  

hidden_layer1 = tf.compat.v1.matmul(x, w1) + b1

w2 = tf.compat.v1.Variable(tf.zeros([32, 64]), name='weight2') 
b2 = tf.compat.v1.Variable(tf.zeros([64]), name='bias2')   

hidden_layer2 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden_layer1, w2) + b2)

w3 = tf.compat.v1.Variable(tf.zeros([64, 128]), name='weight3') 
b3 = tf.compat.v1.Variable(tf.zeros([128]), name='bias3')   

hidden_layer3 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden_layer2, w3) + b3)

# output layer
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([128,3]), name='weight4')
b4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([3]), name='bias4')

hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(hidden_layer3, w4) + b4)

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
# loss = 'categorical_crossentropy'과 동일 

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)   
train = optimizer.minimize(loss)
# train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)  

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
y_predict = sess.run(tf.math.argmax(sess.run(hypothesis, feed_dict={x:x_test}), axis=1)) 
y_test = sess.run(tf.math.argmax(y_test, axis=1))

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)      # acc :  0.3
sess.close()