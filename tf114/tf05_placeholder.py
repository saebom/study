# 텐서1에서 사용하는 문법3 : constant(상수), placeholder(변하지않는놈, 데이터를정의하는놈), variable(변수)

import numpy as np
import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly()) # True

# 즉시실행모드
tf.compat.v1.disable_eager_execution() # 즉시실행모드 꺼!!!
print(tf.executing_eagerly()) # True

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()

############################## 요기서부터 ####################################

a = tf.compat.v1.placeholder(tf.float32)    # placeholder를 정의, 텐서머신에 공간을 만들어 줌
b = tf.compat.v1.placeholder(tf.float32)    # placeholder를 정의, 텐서머신에 공간을 만들어 줌

add_node = a + b
print(sess.run(add_node, feed_dict={a:3, b:4.5}))   # 7.5 feed_dict는 placeholder의 실행함수 
print(sess.run(add_node, feed_dict={a:[1, 3], b:[2, 4]}))   # [3. 7.]

add_node = a + b
print(sess.run(add_node, feed_dict={a:3, b:4.5}))   # 7.5 feed_dict는 placeholder의 실행함수 
print(sess.run(add_node, feed_dict={a:[1, 3], b:[2, 4]}))   # [3. 7.]

add_and_triple = add_node * 3
print(add_and_triple)     # Tensor("mul:0", dtype=float32)
print(sess.run(add_and_triple, feed_dict={a:3, b:4.5}))     # 22.5

