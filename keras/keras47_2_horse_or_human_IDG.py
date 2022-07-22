import numpy as np
from keras.preprocessing.image import ImageDataGenerator


#1. 데이터
train_datagen = ImageDataGenerator(
    rescale = 1./255,       # 스케일 조정
    horizontal_flip= True,  # 수평으로 반전
    vertical_flip=True,     # 수직으로 반전
    width_shift_range=0.1,  # 수평 이동 범위
    height_shift_range=0.1, # 수직 이동 범위
    rotation_range=5,       # 회전 범위
    zoom_range=1.2,         # 확대 범위
    shear_range=0.7,        # 기울기 범위
    fill_mode='nearest',     # 채우기 모드
    featurewise_center=True,
    featurewise_std_normalization=True,  
    #validation 나누기
    validation_split=0.2
    )

test_datagen = ImageDataGenerator(
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

print(train_generator[0][0].shape)      # (600, 150, 150, 3)
print(train_generator[0][1].shape)      # (600, 2)
print(validation_generator[0][0].shape) # (205, 150, 150, 3)
print(validation_generator[0][1].shape) # (205, 2)   


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
