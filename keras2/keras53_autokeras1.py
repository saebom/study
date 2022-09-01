# 오토케라스 공식문서 : https://autokeras.com/
# 나머지는 공식문서에서 찾아서 할 것.
# 그리드서치랑 캐라스 엮으면 컴퓨터 엄청 좋아야한다 아니면 안돌아간다.
# 그래서 오토케라스를 사용해봤다. 
"""
오토케라스란?
Auto-Keras는 자동화된 기계 학습 (AutoML)을 위한 오픈 소스 소프트웨어 라이브러리입니다.
오토케라스 다운로드 pip install autokeras
오토케라스의 모델이 따로 있으므로 공식문서에서 필요할 때 찾아서 쓸 것
 - ImageClassifier 이것도 오토케라스꺼다
 """
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