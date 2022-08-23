import tensorflow as tf
tf.compat.v1.set_random_seed(777)

# [실습]
# 08_2를 카피해서 아래를 맹그러바!!!

#1. 데이터
x_train_data = [1, 2, 3]
y_train_data = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape=[None])    # shape는 input shape, None은 자동으로 잡아줌
y_train = tf.placeholder(tf.float32, shape=[None])    # shape는 input shape, None은 자동으로 잡아줌


W = tf.Variable(tf.random_normal([1]), dtype=tf.float32)    # []안의 숫자는 1 또는 변수의 갯수와 맞춰줘야 함
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)   # 통상적으로 b의 디폴트는 0이다.

#2. 모델 구성  
hypothesis = x_train * W + b  

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))  
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)   
train = optimizer.minimize(loss)


################################# 1. Session() // sees.run(변수) #################################

# 3-2. 훈련
# with tf.compat.v1.Session() as sess:    # with문 안에 작업할 것을 넣어주면 sess.close() 안해도 됨
# # sess = tf.compat.v1.Session()
#     sess.run(tf.global_variables_initializer())     # 변수 초기화

#     epochs = 101
#     for step in range(epochs):
#         # sess.run(train) 
#         _, loss_val, W_val, b_val = sess.run([train, loss, W, b],               # _, 는 train을 반환은 하지 않지만 실행은 시키겠다는 뜻
#                                     feed_dict={x_train:x_train_data, y_train:y_train_data})
#         if step %20 == 0:
#             # print(step, sess.run(loss), sess.run(W), sess.run(b))
#             print(step, loss_val, W_val, b_val)

# #4. 평가, 예측
#     x_text_data = [6, 7, 8]
#     x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
#     y_predict = x_test * W_val + b_val  # y_predict = model.y_predict(x_test)와 동일

#     print('[6, 7, 8] 예측 : ', sess.run(y_predict, feed_dict={x_test:x_text_data}))  # [6, 7, 8] 예측 :  [13.001743 15.002213 17.00268 ]

################################# 2. Session() // 변수.eval(session=sess) #################################
# 3-2. 훈련
# with tf.compat.v1.Session() as sess:    
#     sess.run(tf.global_variables_initializer())   # 변수 초기화

#     epochs = 101
#     for step in range(epochs):
#         _, loss_val, W_val, b_val = sess.run([train, loss, W, b],               # _, 는 train을 반환은 하지 않지만 실행은 시키겠다는 뜻
#                                     feed_dict={x_train:x_train_data, y_train:y_train_data})
#         if step %20 == 0:
#             # print(step, sess.run(loss), sess.run(W), sess.run(b))
#             print(step, loss_val, W_val, b_val)

# #4. 평가, 예측
#     x_text_data = [6, 7, 8]
#     x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
#     y_predict = x_test * W_val + b_val  # y_predict = model.y_predict(x_test)와 동일
  
    # print('[6, 7, 8] 예측 : ', y_predict.eval(session=sess, feed_dict={x_test:x_text_data}))  # [6, 7, 8] 예측 :  [13.001743 15.002213 17.00268 ]


################################# 3. InteractiveSession() // 변수.eval() #################################
# 3-3. 훈련
sess = tf.compat.v1.InteractiveSession()        # InteractiveSession 은 with 문 사용못함        
sess.run(tf.global_variables_initializer())   # 변수 초기화
        
epochs = 101
for step in range(epochs):
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b],               # _, 는 train을 반환은 하지 않지만 실행은 시키겠다는 뜻
                                feed_dict={x_train:x_train_data, y_train:y_train_data})
    if step %20 == 0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        print(step, loss_val, W_val, b_val)

#4. 평가, 예측
x_text_data = [6, 7, 8]
x_test = tf.placeholder(tf.float32)
y_predict = x_test * W_val + b_val  # y_predict = model.y_predict(x_test)와 동일

print('[6, 7, 8] 예측 : ',  y_predict.eval(feed_dict={x_test:x_text_data}))  # [6, 7, 8] 예측 :  [13.001743 15.002213 17.00268 ]
sess.close() 
