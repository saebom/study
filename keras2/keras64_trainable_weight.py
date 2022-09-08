import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))          
model.add(Dense(1))

model.summary()

print(model.weights)    
   
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[-0.25118035,  0.01437318, -0.7196353 ]], dtype=float32)>, 
# <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, 
# <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, 
# numpy= array([[-0.9753833 ,  0.2593081 ],
#        [ 0.93891096,  0.31711078],
#        [ 0.5651659 , -0.8122597 ]], dtype=float32)>, 
# <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, 
# <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, 
# numpy= array([[-0.79742175],[ 0.27166235]], dtype=float32)>, 
# <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>] 

print("=================================================================")
# print(model.trainable_weights)

# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[-0.25118035,  0.01437318, -0.7196353 ]], dtype=float32)>, 
# <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, 
# 
# <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, 
# numpy= array([[-0.9753833 ,  0.2593081 ],
#        [ 0.93891096,  0.31711078],
#        [ 0.5651659 , -0.8122597 ]], dtype=float32)>, 
# <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, 
# <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, 
# 
# numpy= array([[-0.79742175],[ 0.27166235]], dtype=float32)>, 
# <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

print(len(model.weights))           # 6
print(len(model.trainable_weights)) # 6

model.trainable = False

print(len(model.weights))           # 6
print(len(model.trainable_weights)) # 0

print("==============================================================")
print(model.trainable_weights)      # []


model.summary()

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, batch_size=1, epochs=100)

y_predict = model.predict(x)
print(y_predict[:3])
# [[0.61083233]
#  [1.2216647 ]
#  [1.8324974 ]]