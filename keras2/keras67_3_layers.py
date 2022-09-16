from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

#1. 전체 모델의 trainable
# model.trainable = False
# =================================================================
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17
# =================================================================

#2. 레이어의 trainable
# for layer in model.layers:
#     layer.trainable = False
# =================================================================
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17    
# =================================================================

# model.layers[0].trainable = False    # 첫번째 dense 레이어가 훈련 안됨
# model.layers[1].trainable = False    # dense_1 레이어가 훈련 안됨
model.layers[2].trainable = False      # dense_2 레이어가 훈련 안됨

model.summary()

print(model.layers)
# [<keras.layers.core.dense.Dense object at 0x000002346A2F55B0>, 
# <keras.layers.core.dense.Dense object at 0x000002340CA5D820>, 
# <keras.layers.core.dense.Dense object at 0x000002340CA5D7F0>]
