from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D
from keras.applications import VGG16, InceptionV3
from keras.datasets import cifar100

#1. 데이터
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)


# 2. 모델
base_model = InceptionV3(weights = 'imagenet', include_top=False)
# base_model.summary()

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(102, activation='relu')(x)

output1 = Dense(100, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output1)

#1. 
# for layer in base_model.layers:     # base_model.layers[3]  # 특정 레이어를 훈련시키지 않을 수 있음
#     layer.trainable = False

# model.summary()

# Total params: 22,022,082
# Trainable params: 219,298
# Non-trainable params: 21,802,784

#2. 
base_model.trainable = False

# model.summary()
# Total params: 22,022,082
# Trainable params: 219,298
# Non-trainable params: 21,802,784

# print(base_model.layers)
