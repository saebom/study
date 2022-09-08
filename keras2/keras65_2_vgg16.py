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
# block5_pool (MaxPooling2D)   (None, 1, 1, 512)         0
# =================================================================
# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688
# _________________________________________________________________

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))

model.trainable = False     # 모델의 trainable을 시키지 않는다
model.summary()
 
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# vgg16 (Model)                (None, 1, 1, 512)         14714688
# _________________________________________________________________
# flatten (Flatten)            (None, 512)               0
# _________________________________________________________________
# dense (Dense)                (None, 100)               51300
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                1010
# =================================================================
# Total params: 14,766,998
# Trainable params: 14,766,998
# Non-trainable params: 0
# _________________________________________________________________

print(len(model.weights))
print(len(model.trainable_weights))
#                   _____________________________________________________________________
#                            Trainable = True        vgg = False        model = False
#                   =====================================================================
# len(model.weights)                  30                 30                   30
#                   _____________________________________________________________________
# len(model.trainable_weights)        30                 4                     0
#                   =====================================================================
# 

