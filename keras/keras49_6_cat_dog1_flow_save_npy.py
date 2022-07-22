import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import DirectoryIterator
from keras.layers import MaxPooling2D, Dropout
from sympy import Max
import tensorboard 

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
    )

test_datagen = ImageDataGenerator(
    rescale=1./255    
)

xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/cat_dog/train/', 
    target_size=(150, 150), 
    batch_size=8005,          
    class_mode='categorical',   
    color_mode='rgb', 
    shuffle=True,   
)

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/image/cat_dog/test/', 
    target_size=(150, 150),
    batch_size=2023,
    class_mode='categorical',    
    color_mode='rgb',
    shuffle=True,
)

np.save('d:/study_data/_save/_npy/keras49_5_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/keras49_5_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/keras49_5_test_x.npy', arr=xy_test[0][0])
np.save('d:/study_data/_save/_npy/keras49_5_test_y.npy', arr=xy_test[0][1])

'''
#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2, 2), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))  
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))  
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(xy_train[0][0], xy_train[0][1], epochs=30, batch_size=32, 
                 validation_split=0.2, verbose=1)   

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
# loss : 0.047354985028505325
# val_loss : 2.5059962272644043
# accuracy : 0.9800124764442444
# val_accuracy : 0.5459088087081909
# ==================================================================================================
'''