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
x_train = np.load('D:/study_data/_save/_npy/keras49_5_train_x.npy')
y_train = np.load('D:/study_data/_save/_npy/keras49_5_train_y.npy')
x_test = np.load('D:/study_data/_save/_npy/keras49_5_test_x.npy')
y_test = np.load('D:/study_data/_save/_npy/keras49_5_test_y.npy')


#2. 모델
model = Sequential()
input = Input(shape=(50, 60, 3))
rn50 = Xception(weights='imagenet', include_top=False)(input)
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
model.fit(x_train, y_train, epochs=100, validation_split=0.4, verbose=1,
          callbacks=[es], batch_size=64)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)



#=============================== 이전 결과값 ====================================#
# loss : 0.6876140832901001
# val_loss : 1.0849494934082031
# accuracy : 0.5495495200157166
# val_accuracy : 0.5121951103210449
#================================================================================#


#=============================== ResNet50 결과값 ================================#

#================================================================================#
