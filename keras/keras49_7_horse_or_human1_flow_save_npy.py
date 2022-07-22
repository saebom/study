import numpy as np
from keras.preprocessing.image import ImageDataGenerator

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale = 1./255,       
    horizontal_flip= True,  
    vertical_flip=True,    
    width_shift_range=0.1,  
    height_shift_range=0.1, 
    rotation_range=5,       
    zoom_range=1.2,         
    shear_range=0.7,        
    fill_mode='nearest',     
    featurewise_center=True,
    featurewise_std_normalization=True,  
    #validation 나누기
    validation_split=0.2
    )

train_datagen2 = ImageDataGenerator(
    rescale = 1./255,           
    )

train_generator = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/horse_or_human/',
    target_size=(150, 150), 
    batch_size=600,          
    class_mode='binary',   
    color_mode='rgb', 
    shuffle=True,
    subset='training' # set as training data
    # Found 822 images belonging to 2 classes.    
)

validation_generator = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/horse_or_human/', 
    target_size=(150, 150), 
    batch_size=600,          
    class_mode='binary',   
    color_mode='rgb', 
    subset='validation' # set as validation data
    # Found 205 images belonging to 2 classes.   
)

x_train = train_generator[0][0]
y_train = train_generator[0][1]
x_test = validation_generator[0][0]
y_test = validation_generator[0][1]


# 증폭 사이즈
augment_size = 400
batch_size = 2000   # save 파일 저장 시 전체 이미지 갯수로 batch_size 맞춰줌
randidx = np.random.randint(x_train.shape[0], size=augment_size)

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(x_train.shape[0]) # 600
print(randidx)          # [20693 47880 21722 ... 50370 50531 26723]
print(randidx.shape)    # (400,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape) # (400, 150, 150, 3)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 150, 150, 3)
x_train = x_train.reshape(x_train.shape[0], 150, 150, 3)
x_test = x_test.reshape(x_test.shape[0], 150, 150, 3)

x_augmented = train_datagen.flow(x_augmented, y_augmented,
                              batch_size=augment_size,
                              shuffle=False).next()[0]      # next()[0] => x를 넣겠다는 의미
                                                            # shuffle=False이므로 label값 변환없이 들어감. 나중에 y값을 그대로 쓸 수 있음
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape) # (1000, 150, 150, 3) (1000,)

xy_train = train_datagen2.flow(x_train, y_train,
                              batch_size=batch_size,
                              shuffle=False)

xy_test = train_datagen2.flow(x_test, y_test,
                              batch_size=batch_size,
                              shuffle=False)


np.save('d:/study_data/_save/_npy/keras49_5_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/keras49_5_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/keras49_5_test_x.npy', arr=xy_test[0][0])
np.save('d:/study_data/_save/_npy/keras49_5_test_y.npy', arr=xy_test[0][1])


'''
#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), 
                 input_shape=(150,150,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# hist = model.fit(train_generator, epochs=30, batch_size=32, 
#                  validation_data=validation_generator,
#                  validation_steps=4,
#                  )   
hist = model.fit_generator(train_generator, epochs = 20, steps_per_epoch = 1, 
                    validation_data = validation_generator,
                    validation_steps = 4,
                    ) 

#. 평가, 예측
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


# ==================================== fit loss 및 accuracy ========================================
# loss : 0.6876140832901001
# val_loss : 1.0849494934082031
# accuracy : 0.5495495200157166
# val_accuracy : 0.5121951103210449
# ==================================================================================================
'''