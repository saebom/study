from numpy import gradient
import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

#1. 데이터
x_train = [1, 2, 3] 
y_train = [1, 2, 3]
x_test = [4, 5, 6]
y_test = [4, 5, 6]


x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
# w = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='weight')
w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')    # 고정값 10
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run(w))  # [0.80269563]

#2. 모델 구성  
hypothesis = x * w  

loss = tf.reduce_mean(tf.square(hypothesis-y)) 

lr = 0.1
gradient = tf.reduce_mean((w * x - y) * x)
descent = w - lr * gradient
update = w.assign(descent)

w_history = []
loss_history = []

# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    _, loss_v, w_v = sess.run([update, loss, w], 
                              feed_dict={x:x_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)

############################ [실습] R2로 맹그러봐!!!!! #############################
y_predict = x_test * w_v    

print(y_predict)    # [4.00006676 5.00008345 6.00010014]

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)      # r2 :  0.999999989276489

mae = mean_absolute_error(y_test, y_predict)
print('mae : ', mae)    # mae :  8.344650268554688e-05

sess.close()
    