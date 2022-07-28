import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


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

paris_train = train_datagen.flow_from_directory(
    'D:/study_data/_project1/img/fashion_img/',
    target_size=(50, 75), 
    batch_size=600,          
    class_mode='categorical',   
    color_mode='rgb', 
    shuffle=True,
    subset='training' 
    # Found 2837 images belonging to 3 classes.
)

paris_validation = train_datagen.flow_from_directory(
    'D:/study_data/_project1/img/fashion_img/', 
    target_size=(50, 75), 
    batch_size=600,          
    class_mode='categorical',   
    color_mode='rgb', 
    subset='validation' 
    # Found 708 images belonging to 3 classes.
)

# print(paris_train[0][0].shape)  # (600, 100, 150, 3)      
# print(paris_train[0][1].shape)  # (600, 2)    
# print(paris_validation[0][0].shape) # (600, 100, 150, 3)
# print(paris_validation[0][1].shape) # (600, 2)


#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), 
                 input_shape=(100,150,3), activation='relu'))
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
model.add(Dense(2, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# hist = model.fit(train_generator, epochs=30, batch_size=32, 
#                  validation_data=validation_generator,
#                  validation_steps=4,
#                  )   
hist = model.fit_generator(paris_train, epochs = 3, steps_per_epoch = 1, 
                    validation_data = paris_validation,
                    validation_steps = 1,
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



# ==================================== fit loss 및 accuracy ========================================
# loss : 0.6876140832901001
# val_loss : 1.0849494934082031
# accuracy : 0.5495495200157166
# val_accuracy : 0.5121951103210449
# ==================================================================================================
