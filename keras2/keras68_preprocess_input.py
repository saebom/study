from keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np
import warnings
warnings.filterwarnings('ignore')

model = ResNet50(weights='imagenet')
img_path = 'd:/study_data/_data/dog/dog05.jpg'
img = image.load_img(img_path, target_size=(224, 224))
print(img)  # <PIL.Image.Image image mode=RGB size=224x224 at 0x1850A003640>


x = image.img_to_array(img)
print("====================== image.img_to_array(img) ======================")
print(x, '\n', x.shape)     #  (224, 224, 3)

x = np.expand_dims(x, axis=0)   # 4차원으로 변환해야 훈련할 수 있음. 
                                # expand_dims는 차원을 늘려준다는 것. reshape로 사용해도 됨
                                # axis=0은  첫번째 행이 생기고 axis=2는 세번째 행이 생김
print("=================== image.img_to_array(x, axis=0) ====================")
print(x, '\n', x.shape)     #  (1, 224, 224, 3)
print(np.min(x), np.max(x))

x = preprocess_input(x)
print("================ preprocess_input(x) ======================")
print(x, '\n', x.shape)     # (224, 224, 3)
print(np.min(x), np.max(x))


print("====================== model.predict ======================")
preds = model.predict(x)
print(preds, '\n', preds.shape) # (1, 1000)

print("결과는 : ", decode_predictions(preds, top=5)[0])
# 결과는 :  [('n02112018', 'Pomeranian', 0.99524313), 
# ('n02086079', 'Pekinese', 0.00285924), 
# ('n02112137', 'chow', 0.00029959617), 
# ('n02112350', 'keeshond', 0.00024104609), 
# ('n02085620', 'Chihuahua', 0.00015569935)]
