from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D
from keras.applications import VGG16
from keras.datasets import cifar100

# 함수형으로 맹그러봐
#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)


# 2. 모델

model = Sequential()

input = Input(shape=(32, 32, 3))
vgg16 = VGG16(weights='imagenet', include_top=False)(input)
gap = GlobalAveragePooling2D()(vgg16)
hidden1 = Dense(128, activation='relu')(gap)
output = Dense(100, activation='softmax')(hidden1)
model = Model(inputs=input, outputs=output)

model.trainable = False     # 모델의 trainable을 시키지 않는다
model.summary()
 
print(len(model.weights))
print(len(model.trainable_weights))

############################################################
import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
############################################################

#3. 컴파일, 훈련
model.compile(optimizer='adam', metrics=['accuracy'], 
                loss='sparse_categorical_crossentropy')

model.fit(x_train, y_train, epochs=300, validation_split=0.4, verbose=1,
          batch_size=128)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)



