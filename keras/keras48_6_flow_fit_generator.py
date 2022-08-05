from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

train_datagen2 = ImageDataGenerator(
    rescale=1./255  
)

# 증폭 사이즈
augment_size = 40000
batch_size = 64
randidx = np.random.randint(x_train.shape[0], size=augment_size)    # (60000, 40000)
                                                                    # np.random.randint는 랜덤하게 int를 뽑아냄
                                                                    # x_train.shape[0] = 60000
                                                                    # x_train.shape[1] = 28
                                                                    # x_train.shape[2] = 28                                
print(x_train.shape) # (60000, 28, 28)
print(x_train.shape[0]) # (60000)
print(randidx)          # [31720 43951 44299 ... 22547 15575 47042]
print(np.min(randidx), np.max(randidx)) # 3 59999
print(type(randidx))    # <class 'numpy.ndarray'>

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)    # (40000, 28, 28)
print(y_augmented.shape)    # (40000,)


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_augmented = x_augmented.reshape(x_augmented.shape[0], 
                                  x_augmented.shape[1], 
                                  x_augmented.shape[2], 1)

x_augmented = train_datagen.flow(x_augmented, y_augmented,
                              batch_size=augment_size,
                              shuffle=False).next()[0]   # next()[0] => x를 넣겠다는 의미
                                                            # shuffle=False이므로 label값 변환없이 들어감. 나중에 y값을 그대로 쓸 수 있음
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

xy_train = train_datagen2.flow(x_train, y_train,
                              batch_size=batch_size,
                              shuffle=False)

xy_test = train_datagen2.flow(x_test, y_test,
                              batch_size=batch_size,
                              shuffle=False)

#2. 모델
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import *

model = Sequential()
input1 = Input(shape=(28, 28, 1))
dense1 = Conv2D(64, kernel_size=(4, 4), padding='same')(input1)
dense2 = MaxPooling2D(2, 2)(dense1)
dense3 = Dropout(0.2)(dense2)
dense4 = Conv2D(64, (4, 4), padding='valid', activation='relu')(dense3)
dense5 = MaxPooling2D(2, 2)(dense4)         
dense6 = Dropout(0.2)(dense5)
dense7 = Conv2D(64, (4, 4), padding='same', activation='relu')(dense6)
dense8 = MaxPooling2D(2, 2)(dense7)         
dense9 = Dropout(0.2)(dense8)
dense10 = Flatten()(dense9)    # (N, 63)  (N, 175)
dense11 = Dense(128, activation='relu')(dense10)
dense12 = Dropout(0.2)(dense11)
dense13 = Dense(64, activation='relu')(dense12)
output1= Dense(10, activation='softmax')(dense13)
model = Model(inputs=input1, outputs=output1)
model.summary()


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['mse','accuracy'])

hist = model.fit_generator(xy_train, epochs=100, steps_per_epoch=len(xy_train),
                           validation_data=xy_test, validation_steps=4)
# hist = model.fit(x_train, y_train, epochs=100, steps_per_epoch=len(xy_train))

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss :', loss[-1])       
print('val_loss :', val_loss[-1])
print('accuracy :', accuracy[-1])
print('val_accuracy :', val_accuracy[-1])


#그래프로 비교
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path = 'C:\Windows\Fonts\malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.plot(hist.history['accuracy'], marker='.', c='orange', label='accuracy')
plt.plot(hist.history['val_accuracy'], marker='.', c='green', label='val_accuracy')
plt.grid()    
plt.title('로스값과 검증로스값')    
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()   
plt.show()



#=============================== 이전 결과값 ====================================#
# loss :  0.28520235419273376
# accuracy :  0.9143000245094299
#================================================================================#


#=============================== 증폭 후 결과값 ===================================#
# loss : 5.359977990337939e-07
# val_loss : 5.371749693949823e-07
# accuracy : 0.10181249678134918
# val_accuracy : 0.0997999981045723
#================================================================================#
