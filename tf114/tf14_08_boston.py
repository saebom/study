import tensorflow as tf
import numpy as np
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=1234, train_size=0.8
)

x = tf.compat.v1.placeholder(tf.float32, hape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, hape=[None, 1])

w = tf.compat.v1.Variable(tf.zeros([13, 1]), name='weight')    
b = tf.compat.v1.Variable(tf.zeros([1]), name='bias')   

#2. 모델 구성  
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)    # sigmoid 를 씌운다

loss = tf.reduce_mean(tf.square(hypothesis-y)) 

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)   
train = optimizer.minimize(loss)


for step in range(21):
    _, loss_v, w_v = sess.run([update, loss, w], 
                              feed_dict={x:x_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)

############################ [실습] R2로 맹그러봐!!!!! #############################
y_predict = x_test * w_v    

print(y_predict)    # [4.00006676 5.00008345 6.00010014]

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)      # r2 :  0.999999989276489

mae = mean_absolute_error(y_test, y_predict)
print('mae : ', mae)    # mae :  8.344650268554688e-05

sess.close()
