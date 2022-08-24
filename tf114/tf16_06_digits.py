import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
tf.set_random_seed(1234)


#1. 데이터
datasets = load_digits()
x, y = datasets.data, datasets.target.reshape(-1, 1)   
print(x.shape, y.shape) # (1797, 64) (1797, 1)

ohe = OneHotEncoder()
ohe.fit(y)
y = ohe.transform(y).toarray()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=1234, shuffle=True
)
print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(x_train.dtype, y_train.dtype) # float64 float64
print(x_test.dtype, y_test.dtype)   # float64 float64


x = tf.compat.v1.placeholder(tf.float32, shape=[None, 64])
# w = tf.compat.v1.Variable(tf.compat.v1.random_normal([13, 3]), name='weight') # weight는 random_normal이나 zeros를 사용해도 괜찮음
# b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1, 3]), name='bias')  
w = tf.compat.v1.Variable(tf.zeros([64, 10]), name='weight') # weight는 random_normal이나 zeros를 사용해도 괜찮음
b = tf.compat.v1.Variable(tf.zeros([1, 10]), name='bias')  
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

#2. 모델구성 // 시작
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# model.add(Dense(3, activation='softmax', input_dim=4))과 동일

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
# loss = 'categorical_crossentropy'과 동일 

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)   
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
print('acc : ', acc)      # acc :  0.9
sess.close()


