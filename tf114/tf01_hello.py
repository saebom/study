import tensorflow as tf
print(tf.__version__)

# print('hello world')

hello = tf.constant('hello world') # 텐서플로에서 constant는 상수
# print(hello)  # Session 없이는 'hello'라는 구조가 출력됨

# sess = tf.Session() # 텐서플로는 출력할 때 Session을 거쳐야 한다
sess = tf.compat.v1.Session()
print(sess.run(hello))  # 그래프 형식의 데이터를 sess.run에 던져서 출력함


