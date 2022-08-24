import numpy as np
import tensorflow as tf
tf.set_random_seed(777)

#1. 데이터
x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]        # [None, 4]
y_data = [[0,0,1],          # 2
          [0,0,1],
          [0,0,1],
          [0,1,0],          # 1
          [0,1,0],
          [1,0,0],          # 0
          [1,0,0]]          # [None, 3]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4, 3]), name='weight') # weight는 random_normal이나 zeros를 사용해도 괜찮음

b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1, 3]), name='bias')  

y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

#2. 모델구성 // 시작
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# model.add(Dense(3, activation='softmax', input_dim=4))과 동일

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
# loss = 'categorical_crossentropy'과 동일 

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)   
train = optimizer.minimize(loss)
# train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)  

# [실습]
# 맹그러!!
# loss 만 출력해 / 맘 바꼇다. accuracy_score도 출력해

# 3-2. 훈련
sess = tf.compat.v1.Session() 
sess.run(tf.compat.v1.global_variables_initializer())     

epochs = 2001
for step in range(epochs):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train], 
                                    feed_dict={x:x_data, y:y_data})
    if step %20 == 0:
        print(step, 'loss : ', cost_val, '\n', hy_val)      
            
#4. 평가, 예측
y_predict = sess.run(tf.math.argmax(hy_val, axis=1)) 
y_data = sess.run(tf.math.argmax(y_data, axis=1))

from sklearn.metrics import accuracy_score, mean_absolute_error
acc = accuracy_score(y_data, y_predict)
print('acc : ', acc)      # acc :  1.0
sess.close()


