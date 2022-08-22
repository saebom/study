import tensorflow as tf
print(tf.__version__)           # 1.14.0
print(tf.executing_eagerly())   # False

# 즉시 실행모드!!!
# tf.compat.v1.disable_eager_execution()    # 즉, 텐서2에서 disable_eager_execution()으로 즉시 실행모드를 False하여 텐서1으로 실행함
# print(tf.executing_eagerly())   # False 

hello = tf.constant('Hello World')
sess = tf.compat.v1.Session()
print(sess.run(hello))  # b'Hello World'
