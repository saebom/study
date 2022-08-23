import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(777)

#1. 데이터
x = [1, 2, 3]
y = [1, 2, 3]
w = tf.compat.v1.placeholder(tf.float32)

#2. 모델 구성  
hypothesis = x * w  

loss = tf.reduce_mean(tf.square(hypothesis-y))  

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={w:curr_w})
        w_history.append((curr_w))
        loss_history.append(curr_loss)

print("===================== W history =========================")
print(w_history)
print("===================== loss history =========================")
print(loss_history)

plt.plot(w_history, loss_history)
plt.xlabel('Weight')
plt.ylabel('Loss')
plt.show()



# ======================== 결과 ================================
# 100 1.2263897e-06 [2.001276] [0.9972493]
# [6, 7, 8] 예측 :  [13.004906 15.006182 17.007458]
# ==============================================================

