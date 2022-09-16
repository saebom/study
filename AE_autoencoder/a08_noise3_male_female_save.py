import numpy as np
from keras.preprocessing.image import ImageDataGenerator


#1. 데이터

test_datagen = ImageDataGenerator(
    rescale = 1./255,           
    )

train_generator = test_datagen.flow_from_directory(
    'd:/study_data/_data/image/men_women/',
    target_size=(150, 150), 
    batch_size=2648,          
    class_mode='categorical',   
    color_mode='rgb', 
    shuffle=True,
    subset='training' # set as training data
    # Found 2648 images belonging to 2 classes.    
)

validation_generator = test_datagen.flow_from_directory(
    'd:/study_data/_data/image/men_women/', 
    target_size=(150, 150), 
    batch_size=661,          
    class_mode='categorical',   
    color_mode='rgb', 
    subset='validation' # set as validation data
    # Found 661 images belonging to 2 classes.   
)

print(train_generator[0][0].shape)      # (600, 150, 150, 3)
print(train_generator[0][1].shape)      # (600, 2)
print(validation_generator[0][0].shape) # (205, 150, 150, 3)
print(validation_generator[0][1].shape) # (205, 2)   


img_generator = test_datagen.flow_from_directory(
    'd:/study_data/_data/me/', 
    target_size=(150, 150),
    batch_size=1,
    class_mode='binary',    
    color_mode='rgb',
    shuffle=True,
)


np.save('d:/study_data/_save/_npy/men_women_train_x.npy', arr=train_generator[0][0])
np.save('d:/study_data/_save/_npy/men_women_train_y.npy', arr=train_generator[0][1])
np.save('d:/study_data/_save/_npy/men_women_test_x.npy', arr=validation_generator[0][0])
np.save('d:/study_data/_save/_npy/men_women_test_y.npy', arr=validation_generator[0][1])
np.save('d:/study_data/_save/_npy/me_x.npy', arr=img_generator[0][0])