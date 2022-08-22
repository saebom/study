import tensorflow as tf
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32)
y = tf.Variable([3], dtype=tf.float32)

init = tf.compat.v1.global_variables_initializer()  # 변수의 초기화를 인정하는 함수
sess.run(init)

print(sess.run(x+y))

