# y = wx + b

import tensorflow as tf
tf.set_random_seed(123)

#1. 데이터
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

W = tf.Variable(33, dtype=tf.float32)
b = tf.Variable(11, dtype=tf.float32)

############################ 완성하시오!!! ##################################

#2. 모델 구성  
hypothesis = x * W + b  # y = wx + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  # mse 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03)   # 경사하강법
train = optimizer.minimize(loss)
# model.compile이다!!!! model.compile(loss='mse, optimzer='sgd') 와 동일함

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 2004
for step in range(epochs):
    sess.run(train) # model.fit 이다!!!! train = optimizer.minimize(tf.reduce_mean(tf.square((x * W + b) - y)))
    if step %20 == 0:
        print(step, sess.run(loss), sess.run(W), sess.run(b))
            # ...
            # 1960  8.007689e-06    0.99671334   0.007471312
            # 1980  7.2730195e-06   0.99686784   0.007120189
            # 2000  6.6050598e-06   0.99701506   0.006785556
        
sess.close() # session을 메모리에서 close함


