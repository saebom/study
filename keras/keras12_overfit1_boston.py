# 2022.07.01 :: 과적합
from tabnanny import verbose
import numpy as ny
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

'''
print(x)
print(y)   # y는 보스턴 집 값
print(x.shape, y.shape)   # (506, 13), (506,)

print(datasets.feature_names)    # feature_name은 사이킷런에서 예제용으로 제공된 데이터의 이름 
print(datasets.DESCR)
'''

#[실습] 아래를 완성할 것
#1) train 0.7
#2) R2를 0.8 이상

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.8, shuffle=True, random_state=66
)

#2. 모델 구성
model = Sequential()
model.add(Dense(7, input_dim=13))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(1))


#3. 훈련
model.compile(loss='mse', optimizer='adam')  

start_time =time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=10, 
                 validation_split=0.2,
                 verbose=1) 
end_time =time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print("=================================================================")   
print(hist)     # <tensorflow.python.keras.callbacks.History object at 0x000002664FF27AF0>
print("=================================================================")
print(hist.history)     # loss 와 val_loss의 key, value를 합쳐놓은 것
print("=================================================================")
print(hist.history['loss'])
print("=================================================================")
print(hist.history['val_loss'])


#4-1. 시각화
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


'''=============================================================
verbose는 학습진행상황 보여주기

verbose 0 걸린시간 : 2.2564609050750732 / 출력없음
verbose 1 걸린시간 : 2.401794672012329  / 보여줌
verbose 2 걸린시간 : 2.2612545490264893 / 프로그래스바 없음
verbose 3 걸린시간 : 2.1730453968048096  / epoch만 나옴 
verbose 4 걸린시간 : 2.201508045196533  / epoch 횟수만 나옴 

================================================================='''