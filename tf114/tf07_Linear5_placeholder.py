import tensorflow as tf
# tf.set_random_seed(123)

#1. 데이터
# x = [1, 2, 3, 4, 5]
# y = [1, 2, 3, 4, 5]
x = tf.placeholder(tf.float32, shape=[None])    # shape는 input shape, None은 자동으로 잡아줌
y = tf.placeholder(tf.float32, shape=[None])    # shape는 input shape, None은 자동으로 잡아줌

# W = tf.Variable(33, dtype=tf.float32) 
# b = tf.Variable(11, dtype=tf.float32)
W = tf.Variable(tf.random_normal([1]), dtype=tf.float32)    # []안의 숫자는 1 또는 변수의 갯수와 맞춰줘야 함
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)    # 통상적으로 b의 디폴트는 0이다.

#2. 모델 구성  
hypothesis = x * W + b  

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)   
train = optimizer.minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:    # with문 안에 작업할 것을 넣어주면 sess.close() 안해도 됨
# sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())     # 변수 초기화

    epochs = 2001
    for step in range(epochs):
        # sess.run(train) 
        _, loss_val, W_val, b_val = sess.run([train, loss, W, b],               # _, 는 train을 반환은 하지 않지만 실행은 시키겠다는 뜻
                                    feed_dict={x:[1,2,3,4,5,], y:[1,2,3,4,5]})
        if step %20 == 0:
            # print(step, sess.run(loss), sess.run(W), sess.run(b))
            print(step, loss_val, W_val, b_val)
            
# sess.close() # session을 메모리에서 close함

