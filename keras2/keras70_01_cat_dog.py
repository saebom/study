import tensorflow as tf
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.layers import Input
from keras.callbacks import EarlyStopping
from keras.applications import VGG16, Xception, ResNet50
import warnings
warnings.filterwarnings('ignore')



#1. 데이터 로드
x_train = np.load('d:/study_data/_save/_npy/keras49_5_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_5_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_5_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_5_test_y.npy')


#2. 모델
model = Sequential()
input = Input(shape=(150, 150, 3))
rn50 = ResNet50(weights='imagenet', include_top=False)(input)
gap = GlobalAveragePooling2D()(rn50)
hidden1 = Dense(128, activation='relu')(gap)
hidden2 = Dense(128, activation='relu')(hidden1)
output = Dense(100, activation='softmax')(hidden2)
model = Model(inputs=input, outputs=output)

# vgg16.trainable = False     
# model.trainable = False     

#3. 컴파일, 훈련
model.compile(optimizer='adam', metrics=['accuracy'], 
                loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                   restore_best_weights=True, verbose=1)
model.fit(x_train, y_train, epochs=300, validation_split=0.4, verbose=1,
          callbacks=[es], batch_size=64)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)



#=============================== 이전 결과값 ====================================#
# loss : 0.047354985028505325
# val_loss : 2.5059962272644043
# accuracy : 0.9800124764442444
# val_accuracy : 0.5459088087081909
#================================================================================#


#=============================== ResNet50 결과값 ================================#
# loss :  0.6931127905845642
# acc :  0.5121951103210449
#================================================================================#
