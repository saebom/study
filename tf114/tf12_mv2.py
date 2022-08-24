import tensorflow as tf
tf.compat.v1.set_random_seed(2022)

x_data = [[73, 51, 65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]                         # (5, 3)
y_data = [[152], [185], [180], [205], [142]]    # (5, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1]), name='weight') # bias는 통상 0이므로 1로 설정해줌
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')

hypothesis = tf.compat.v1.matmul(x, w) + b


################################### [실습] 맹그러 ###################################

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=8e-5)   
train = optimizer.minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:    
    sess.run(tf.compat.v1.global_variables_initializer())     

    epochs = 2001
    for step in range(epochs):
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train], 
                                       feed_dict={x:x_data, y:y_data})
        if step %20 == 0:
            print(step, 'loss : ', cost_val, '\n', hy_val)      
            
#4. 평가, 예측
from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data, hy_val)
print('r2 : ', r2)      # r2 :  0.43988023342202087

mae = mean_absolute_error(y_data, hy_val)
print('mae : ', mae)    # mae :  15.167121887207031
sess.close()   




