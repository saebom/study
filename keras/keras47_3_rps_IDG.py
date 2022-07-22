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

test_datagen = ImageDataGenerator(
    rescale = 1./255,           
    )

train_generator = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/rps',
    target_size=(150, 150), 
    batch_size=2020,          
    class_mode='categorical',   
    color_mode='rgb', 
    shuffle=True,
    subset='training' # set as training data
    # Found 2016 images belonging to 3 classes.      
)

validation_generator = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/rps/', 
    target_size=(150, 150), 
    batch_size=600,          
    class_mode='categorical',   
    color_mode='rgb', 
    subset='validation' # set as validation data
    # Found 504 images belonging to 3 classes. 
)

print(train_generator[0][0].shape)      # (2016, 150, 150, 3)
print(train_generator[0][1].shape)      # (2016, 3)
print(validation_generator[0][0].shape) # (504, 150, 150, 3)
print(validation_generator[0][1].shape) # (504, 3)   


#2. 모델
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
# from tensorflow.keras.models import Sequential
# from keras.layers import *

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
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# hist = model.fit(train_generator, epochs=10, steps_per_epoch = 1, 
#                  validation_data=validation_generator,
#                  validation_steps=4,
#                  )   
hist = model.fit_generator(train_generator, epochs = 20, steps_per_epoch = 1, 
                    validation_data = validation_generator,
                    validation_steps = 1,) 

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss :', loss[-1])       
print('val_loss :', val_loss[-1])
print('accuracy :', accuracy[-1])
print('val_accuracy :', val_accuracy[-1])


# #그래프로 비교
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
