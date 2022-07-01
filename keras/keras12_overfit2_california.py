import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

'''
print(x)
print(y)
print(x.shape, y.shape)  # (20640, 8) (206040,)

print(datasets.feature_names)
print(datasets.DESCR)
'''

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=0
)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))


#3. 훈련
model.compile(loss='mse', optimizer='adam')   

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=90, batch_size=100, 
          validation_split=0.2,
          verbose=1)
end_time = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)      # 0.7170546650886536

'''
print("=================================================================")   
print(hist)     # <tensorflow.python.keras.callbacks.History object at 0x000002664FF27AF0>
print("=================================================================")
print(hist.history)     # loss 와 val_loss의 key, value를 합쳐놓은 것
print("=================================================================")
print(hist.history['loss'])
print("=================================================================")
print(hist.history['val_loss'])
'''

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = 'C:\Windows\Fonts\malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
# plt.title('loss & val_loss')    
plt.title('로스값과 검증로스값')   
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right')   # 우측상단에 라벨표시
plt.legend()   # 자동으로 빈 공간에 라벨표시
plt.show()

# y_predict = model.predict(x_test)  

# r2 = r2_score(y_test, y_predict)
# print('r2 스코어: ', r2)  

# loss :  0.9029045104980469

