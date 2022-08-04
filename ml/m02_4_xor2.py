import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]   
y_data = [0, 1, 1, 0]  # => XOR 게이트 (같으면 0, 다르면 1)

#2. 모델
# model = LinearSVC()
# model = Perceptron()
model = SVC()   # MLP(multi layer perceptron)

#3. 훈련
model.fit(x_data, y_data)

#4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과 : ", y_predict)

results = model.score(x_data, y_data)
print('model.score : ', results)

acc = accuracy_score(y_data, y_predict)
print('accuracy_score : ', acc)


#====================================  결과 =======================================#
# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [0 1 1 0]
# model.score :  1.0
# accuracy_score :  1.0
#==================================================================================#

