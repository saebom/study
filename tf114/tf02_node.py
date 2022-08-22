import tensorflow as tf

node1 = tf.constant(3.0, tf.float32) # 자료형까지 명시해야 함
node2 = tf.constant(4.0)    # 기본형이므로 float 등 자료형 생략해도 됨
# node3 = node1 + node2
node3 = tf.add(node1, node2)
# print(node3)    # Tensor("add:0", shape=(), dtype=float32)

sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(node3))  # 7.0



