import autokeras as ak
import time
import keras
import tensorflow as tf
from keras.datasets import mnist
print(ak.__version__)   # 1.0.20

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

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