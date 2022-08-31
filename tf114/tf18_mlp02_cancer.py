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

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([30, 100]), name='weight1') 
b1 = tf.compat.v1.Variable(tf.zeros([100]), name='bias1')   

hidden_layer1 = tf.compat.v1.matmul(x, w1) + b1

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100, 100]), name='weight2') 
b2 = tf.compat.v1.Variable(tf.zeros([100]), name='bias2')   

hidden_layer2 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden_layer1, w2) + b2)
# hidden_layer2 = tf.compat.v1.matmul(hidden_layer1, w2) + b2

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100, 100]), name='weight3') 
b3 = tf.compat.v1.Variable(tf.zeros([100]), name='bias3')   

hidden_layer3 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(hidden_layer1, w3) + b3)
# hidden_layer3 = tf.compat.v1.matmul(hidden_layer2, w3) + b3

# output layer
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,1]), name='weight4')
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias4')

hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(hidden_layer3, w4) + b4)

#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  # binary_crossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)   
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
y_predict = sess.run(tf.cast(hy_val>0.5, dtype=tf.float32))   
from sklearn.metrics import accuracy_score, mean_absolute_error
acc = accuracy_score(y_train, y_predict)
print('acc : ', acc)      # acc :  0.6263736263736264
mae = mean_absolute_error(y_train, hy_val)
print('mae : ', mae)      # mae :  0.4680594074857104
sess.close()

