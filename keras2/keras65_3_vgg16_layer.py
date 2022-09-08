import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

# model = VGG16()
vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))

# vgg16.summary()
# vgg16.trainable = False     # 가중치를 동결시킨다!!! vgg16의 trainable을 시키지 않는다
# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))

# model.trainable = False     # 모델의 trainable을 시키지 않는다
model.summary()
 
print(len(model.weights))
print(len(model.trainable_weights))

################################ keras65_2번 소스에서 아래만 추가 (layer를 확인할 수 있음) ##########################################
print(model.layers)

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
#                                                                      Layer Type Layer Name  Layer Trainable
# 0  <tensorflow.python.keras.engine.training.Model object at 0x000002BFDAAA2F60>  vgg16      True
# 1  <tensorflow.python.keras.layers.core.Flatten object at 0x000002BFE68A5588>    flatten    True
# 2  <tensorflow.python.keras.layers.core.Dense object at 0x000002BFDE2F8D30>      dense      True
# 3  <tensorflow.python.keras.layers.core.Dense object at 0x000002BFDAABDB70>      dense_1    True
