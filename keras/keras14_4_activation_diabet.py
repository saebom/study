import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn import metrics
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import time
from matplotlib import font_manager, rc

datasets = load_diabetes()
x = datasets.data
y = datasets.target

'''
print(x)
print(y)
print(x.shape, y.shape)   # (442, 10) (442,)
print(datasets.feature_names)
print(datasets.DESCR)
'''
# [실습] train_size 조절 가능, R2 0.62 이상

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=72
)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=10))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', 
                              verbose=1, 
                              restore_best_weights=True)
start_time = time.time() 
hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  


# print("=================================================================")   
# print(hist)     # <tensorflow.python.keras.callbacks.History object at 0x000002664FF27AF0>
# print("=================================================================")
# print(hist.history)     # loss 와 val_loss의 key, value를 합쳐놓은 것
# print("=================================================================")
# print(hist.history['loss'])
# print("=================================================================")
# print(hist.history['val_loss'])


#4_1. 그래프로 비교
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



#================================ 적용 전 데이터 ===================================#
# loss :  2692.134033203125
# r2 스코어:  0.5888314290057607
#==================================================================================#

#================================ 적용 후 데이터 ===================================#
# random_state = 72
# loss :  2191.158447265625
# mse :  2191.158447265625
# r2 스코어:  0.6681681298699478
#==================================================================================#

##=============================== 내 용 정 리 ======================================##
# 1) 회귀모델이므로 loss='mae', metrics=['mse'] 사용
#   (회귀모델에서는 'accuracy'= 0이 나옴)
# 2) 회귀모델이므로 accuracy_score 가 아니라 r2 값 사용
# 주의!!! output 레이어에 'sigmoid'가 아니라 'linear' 사용해야 함(분류모델 아니까!!!)
##=================================================================================##
         
##================================ 느 낀 점 ========================================##
# 히든레이어에 activation ='relu'를 적용하니 확실히 값이 향상되었음
# 그러나 드라마틱하게 점수가 높아지는 것은 아님
# 이쯤되니 relu 친구들이 궁금해짐
##=================================================================================##
