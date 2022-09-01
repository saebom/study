import autokeras as ak
import time
import keras
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from sklearn.preprocessing import RobustScaler
print(ak.__version__)   # 1.0.20

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

# loss의 스케일 조정을 위해 0 ~ 255 -> 0 ~ 1 범위로 만들어줌
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

mean = np.mean(x_train, axis=(0 , 1 , 2 , 3))
std = np.std(x_train, axis=(0 , 1 , 2 , 3))
x_train = (x_train-mean)/std
x_test = (x_test-mean)/std

#2. 모델
model = ak.ImageClassifier(
    overwrite=True, 
    max_trials=2
)

#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train, epochs=5)
end_time = time.time() - start_time

#4. 평가, 예측
y_predict = model.predict(x_test)

results = model.evaluate(x_test, y_test)
print("결과 : ", results)
print("걸린시간 : ", round(end_time, 4))

# ======================================================== #
# 결과 :  [0.03811722248792648, 0.9876000285148621]
# 걸린시간 :  4146.4615
# ========================================================= #