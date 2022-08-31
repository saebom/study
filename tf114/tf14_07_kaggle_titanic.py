import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
tf.compat.v1.set_random_seed(1234)


#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'gender_submission.csv')

# 데이터 전처리
train_set[['Pclass', 'Survived']].groupby(['Pclass'], 
                                          as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['Sex', 'Survived']].groupby(['Sex'], 
                                       as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['SibSp', 'Survived']].groupby(['SibSp'], 
                                         as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['Parch', 'Survived']].groupby(['Parch'], 
                                         as_index=False).mean().sort_values(by='Survived', ascending=False)


# Ticket, Cabin, Name 삭제
train_set = train_set.drop(['Ticket', 'Cabin'], axis=1)
test_set = test_set.drop(['Ticket', 'Cabin'], axis=1)

train_set = train_set.drop(['Name'], axis=1)
test_set = test_set.drop(['Name'], axis=1)
# print('train.shape : ', train_set.shape, 'test.shape : ', test_set.shape) #(891, 8) (418, 7)

# Embarked, Sex Object => float 변환
train_set['Embarked'] = train_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(float)
test_set['Embarked'] = test_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(float)

train_set['Sex'] = train_set['Sex'].map({'male':0, 'female':1}).astype(float)
test_set['Sex'] = test_set['Sex'].map({'male':0, 'female':1}).astype(float)

train_set = train_set.drop(['Parch'], axis=1)
print(train_set.columns)    # Index(['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked'], dtype='object')

# x, y 데이터
x = train_set.drop(['Survived'], axis=1)
print(x.shape)  # (891, 6)
print(x.columns)    # 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'

y = train_set['Survived']
print(y)
print(y.shape)  # (891,)

# 라벨인코딩
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123)

print(type(x_train), type(y_train))
print(x_train.dtype, y_train.dtype) # float64 int32
print(x_test.dtype, y_test.dtype)   # float64 int32

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 6])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([6,1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')
hypothesis = tf.compat.v1.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-6)   
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session() 
sess.run(tf.compat.v1.global_variables_initializer())     

epochs = 2001
for step in range(epochs):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train], 
                                    feed_dict={x:x_train, y:y_train})
    if step %20 == 0:
        print(step, 'loss : ', cost_val, '\n', hy_val)      
            
#4. 평가, 예측
y_predict = sess.run(hypothesis, feed_dict={x:x_test})
print(y_predict)   

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)      # r2 :  0.5650995686049121

mae = mean_absolute_error(y_test, y_predict)
print('mae : ', mae)    # mae :  42.928504086612314

sess.close()
    
