import tensorflow as tf
# tf.set_random_seed(123)

#1. 데이터
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

# W = tf.Variable(33, dtype=tf.float32) 
# b = tf.Variable(11, dtype=tf.float32)
# W = tf.Variable(tf.random_normal([1]), dtype=tf.float32)    # 변수가 랜덤으로 들어갈 수 있도록 바꿔보자. []안의 숫자는 입력되는 수
# b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)    # 통상적으로 b의 디폴트는 0이다.
W = tf.Variable(tf.random_uniform([1]), dtype=tf.float32)    # 변수가 랜덤으로 들어갈 수 있도록 바꿔보자. []안의 숫자는 입력되는 수
b = tf.Variable(tf.random_uniform([1]), dtype=tf.float32)


sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(W))  # 첫 번째 랜덤 값 [-1.5080816], 두 번째 랜덤 값[0.07588947], 세 번째 랜던 값 [0.21714649]
print(sess.run(b))  # 첫 번째 랜덤 값 [-1.5080816], 두 번째 랜덤 값[0.07588947], 세 번째 랜던 값 [0.21714649]

#2. 모델 구성  
hypothesis = x * W + b  # y = wx + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  # mse 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03)   # 경사하강법
train = optimizer.minimize(loss)
# model.compile이다!!!! model.compile(loss='mse, optimzer='sgd') 와 동일함

# 3-2. 훈련
with tf.compat.v1.Session() as sess:    # with문 안에 작업할 것을 넣어주면 sess.close() 안해도 됨
# sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 2004

    for step in range(epochs):
        sess.run(train) # model.fit 이다!!!! train = optimizer.minimize(tf.reduce_mean(tf.square((x * W + b) - y)))
        if step %20 == 0:
            print(step, sess.run(loss), sess.run(W), sess.run(b))

# sess.close() # session을 메모리에서 close함


