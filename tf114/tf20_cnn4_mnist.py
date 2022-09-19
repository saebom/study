import tensorflow as tf
import numpy as np
from torch import dropout

tf.compat.v1.set_random_seed(123)

# 즉시 실행모드!!!
tf.compat.v1.disable_eager_execution()    

#1. 데이터
# pip install keras==2.3
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.  
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.  

#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1]) 
y = tf.compat.v1.placeholder(tf.float32, [None, 10])
dropout = tf.compat.v1.placeholder(tf.float32)

# Layer1
w1 = tf.compat.v1.get_variable('w1', shape=[4, 4, 1, 32])  
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')                                   
L1 = tf.nn.relu(L1)                                                 # relu 적용
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding='SAME')
# model.add(Conv2d(64, kernel_size=(2,2), input_shape=(28,28,1), activation='relu')) 과 동일

print(w1)   # <tf.Variable 'w1:0' shape=(4, 4, 1, 32) dtype=float32>
print(L1)   # Tensor("Relu:0", shape=(None, 28, 28, 32), dtype=float32)
print(L1_maxpool)   # Tensor("MaxPool2d:0", shape=(None, 14, 14, 32), dtype=float32)

# Layer2
w2 = tf.compat.v1.get_variable('w2', shape=[4, 4, 32, 64])  
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1, 1, 1, 1], padding='VALID')                                   
L2 = tf.nn.selu(L2)                                                 # selu 적용
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding='SAME')

print(L2)   # Tensor("Selu:0", shape=(None, 11, 11, 64), dtype=float32)
print(L2_maxpool)   # Tensor("MaxPool2d_1:0", shape=(None, 6, 6, 64), dtype=float32)

# Layer3
w3 = tf.compat.v1.get_variable('w3', shape=[4, 4, 64, 64])  
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1, 1, 1, 1], padding='VALID')                                   
L3 = tf.nn.elu(L3)                                                  # elu 적용

print(L3)   # Tensor("Elu:0", shape=(None, 3, 3, 64), dtype=float32)

# Flatten
L_flat = tf.reshape(L3, [-1, 3*3*64])
print("플래튼 : ", L_flat)  # 플래튼 :  Tensor("Reshape:0", shape=(None, 1024), dtype=float32)

# Layer4 DNN
w4 = tf.compat.v1.get_variable('w4', shape=[3*3*64, 128],
                     initializer=tf.compat.v1.keras.initializers.glorot_normal())  # initializer ==> 가중치 초기화 
b4 = tf.Variable(tf.compat.v1.random_normal([128]), name='b4')
L4 = tf.nn.selu(tf.compat.v1.matmul(L_flat, w4) + b4)
# L4 = tf.nn.dropout(L4, rate = 0.3)   # rate = 0.3

# Layer5 DNN
w5 = tf.compat.v1.get_variable('w5', shape=[128, 10],
                     initializer=tf.compat.v1.keras.initializers.glorot_normal())  # initializer ==> 가중치 초기화 
b5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10]), name='b5')
L5 = tf.matmul(L4, w5) + b5

hypothesis = tf.compat.v1.nn.softmax(L5)

print(hypothesis)   # Tensor("Softmax:0", shape=(None, 10), dtype=float32)

# 3-1. 컴파일
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))  
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.compat.v1.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)   

sess = tf.compat.v1.Session() 
sess.run(tf.compat.v1.global_variables_initializer())     

# 3-2. 훈련
training_epochs = 30
batch_size = 100
total_batch = int(len(x_train)/batch_size)  # 60000/100 = 600
print(total_batch)  # 600

for epoch in range(training_epochs):    # 총 30번 돈다
    avg_loss = 0
    for i in range(total_batch):    # 총 600번 돈다
        start = i * batch_size      # 0
        end = start + batch_size    # 100
        batch_x, batch_y = x_train[start:end], y_train[start:end]   # 0~100
        
        feed_dict = {x:batch_x, y:batch_y, dropout: 0.2}
        
        batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        
        avg_loss += batch_loss / total_batch   
    prediction = tf.equal(tf.compat.v1.arg_max(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x:x_test, y:y_test, dropout: 0.0})   
    print('Epoch : ', '%04d' %(epoch + 1), 'loss : {:.9f}'.format(avg_loss),
          'ACC : ', result)    # 1 epoch 값의 loss가 30번마다 출력  
    
print('훈련 끗!!!')        
            
#4. 평가, 예측
# y_predict = sess.run(tf.math.argmax(sess.run(hypothesis, feed_dict={x:x_test}), axis=1)) 
# y_test = sess.run(tf.math.argmax(y_test, axis=1))

# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, y_predict)
# print('acc : ', acc)      # acc :  0.9903

# [실습] verbose에 acc 너줘라

prediction = tf.equal(tf.compat.v1.arg_max(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('ACC : ', sess.run(accuracy, feed_dict={x:x_test, y:y_test, dropout:0.0}))    # ACC :  0.9893

