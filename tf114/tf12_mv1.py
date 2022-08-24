import tensorflow as tf
tf.compat.v1.set_random_seed(777)


#1. 데이터

x1_data = [73., 93., 89., 96., 73. ]    # 국어
x2_data = [80., 88., 91., 98., 66. ]    # 수학
x3_data = [75., 93., 90., 100., 70. ]   # 영어
y_data = [152., 185., 180., 196., 142. ]    # 환산점수

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32) 
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32) 
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32) 
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32) 

#2. 모델
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

################################### [실습] 맹그러 ###################################
#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)   
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)   
train = optimizer.minimize(loss)

# 3-2. 훈련
loss_val_list = []
W_val_list = []

with tf.compat.v1.Session() as sess:    
    
    sess.run(tf.compat.v1.global_variables_initializer())     

    epochs = 2001
    for step in range(epochs):
                # _, loss_val, w1_val, w2_val, w3_val, b_val = sess.run([train, loss, w1, w2, w3, b],              
        #                             feed_dict={x1:x1_data, x2:x2_data, x3:x3_data,
        #                                        y:y_data})
        # if step %20 == 0:
        #     print(step, loss_val, w1_val,w2_val, w3_val,b_val)
        #     loss_val_list.append(loss_val)            
        #     W_val_list.append([w1_val, w2_val, w3_val])
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train], 
                                       feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data} )
        if step %20 == 0:
            print(step, 'loss : ', cost_val, '\n', hy_val)
        
            
#4. 평가, 예측
# y_predict = x1_data *w1_val + x2_data*w2_val + x2_data*w3_val
# print(y_predict)    # [148.40236011 188.5447899  180.68676576 194.89225524 147.88356327]

# from sklearn.metrics import r2_score, mean_absolute_error
# r2 = r2_score(y_data, y_predict)
# print('r2 : ', r2)      # r2 :  0.9985568870603315

# mae = mean_absolute_error(y_data, y_predict)
# print('mae : ', mae)    # mae :  0.6549328416585922
# y_predict = x_test*hy_val
# print(y_predict)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data, hy_val)
print('r2 : ', r2)      # r2 :  0.988362615777223

mae = mean_absolute_error(y_data, hy_val)
print('mae : ', mae)    # mae :  1.7450347900390626
sess.close()   



