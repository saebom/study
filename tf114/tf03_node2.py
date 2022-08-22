import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

# 실습
# 덧셈 node3
# 뺄셈 node4
# 곱셈 node5
# 나눗셈 node6

# 맹그러

# 덧셈
# node3 = node1 + node2
node3 = tf.add(node1, node2)

# 뺄셈 
# node4 = node1 - node2
node4 = tf.subtract(node1, node2)

# 곱셈
# node5 = node1 * node2
node5 = tf.multiply(node1, node2)

# 나눗셈b divide(mod)
# node6 = node1/node2
node6 = tf.divide(node1, node2)

sess = tf.compat.v1.Session()
print(sess.run(node3))  # 5.0
print(sess.run(node4))  # -1.0
print(sess.run(node5))  # 6.0
print(sess.run(node6))  # 0.6666667

