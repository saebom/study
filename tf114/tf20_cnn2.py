import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(123)

#1. 데이터
x_train = np.array([[[[1], [2], [3]],
                     [[4], [5], [6]],
                     [[7], [8], [9]]]])

print(x_train.shape) # (1, 3, 3, 1) 3X3짜리 1개

x = tf.compat.v1.placeholder(tf.float32, [None, 3, 3, 1])

w = tf.compat.v1.constant([[[[1.]], [[1.]]],
                            [[[1.]], [[1.]]]])  # conv2d에서 weight 는 커널사이즈!!!
print(w)    # Tensor("Const:0", shape=(2, 2, 1, 1), dtype=float32)

# L1 = tf.nn.conv2d(x, w, strides=(1,1,1,1), padding='VALID')
# print(L1)   # Tensor("Conv2D:0", shape=(?, 2, 2, 1), dtype=float32)
L1 = tf.nn.conv2d(x, w, strides=(1,2,2,1), padding='SAME')

sess = tf.compat.v1.Session()
output = sess.run(L1, feed_dict={x:x_train})

print(" ============================ 결과 ================================")
print(output)   # [[[[12.], [16.]]
                #   [[24.], [28.]]]] => strides=(1,1,1,1), padding='VALID'
                
                # [[[[12.], [16.], [ 9.]]
                #   [[24.], [28.], [15.]]
                #   [[15.], [17.], [ 9.]]]] => strides=(1,1,1,1), padding='SAME'
                
                # [[[[12.], [ 9.]]
                #   [[15.], [ 9.]]]] => strides=(1,2,2,1), padding='SAME'
print(" ========================= 결과.shape ==============================")
print(output.shape) # (1, 2, 2, 1)
                    # (1, 3, 3, 1)
                    # (1, 2, 2, 1)


    
