import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import time


#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)

print(train_set)
print(train_set.shape)  # (1459, 10)

test_set = pd.read_csv(path + 'test.csv',   # 예측에서 사용!!
                       index_col=0)

print(test_set)
print(test_set.shape)   # (715, 9)  => count 제외한 9개 :: 결과적으로 count 값을 제출

print(train_set.columns)
print(train_set.info())
print(train_set.describe())     # pandas api 에서는 info(), describe() 제공 

#### 결측치 처리 ####
print(train_set.isnull().sum()) # null의 합계 출력
print(test_set.isnull().sum())  # null의 합계 출력
# trian_set = train_set.replace(np.nan, 0, inplace=True)  # nan 대신에 0을 넣음
test_set = test_set.fillna(method='ffill')

train_set = train_set.dropna()  # nan 값 삭제
print(train_set.isnull().sum()) # null의 합계 출력
print(train_set.shape)   # (1328, 10)

x = train_set.drop(['count'], axis=1)

print(x)
print(x.columns)
# print(x.shape)  # (1459, 9)

y = train_set['count']
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=31
)

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation = 'linear', input_dim=9))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss = 'mae', optimizer="adam", metrics=['mse'])

earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='min',
                              restore_best_weights=True,
                              verbose=1)
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=5000, batch_size=100, 
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)
end_time = time.time() - start_time

#4. 평가예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)  

#### mse를 rmse로 변환 ####
y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape)   # (715, 1)

submission = pd.read_csv(path + 'submission.csv')
submission['count'] = y_summit
submission.to_csv(path + 'submission1.csv', index=False)


# print("=================================================================")   
# print(hist)     # <tensorflow.python.keras.callbacks.History object at 0x000002664FF27AF0>
# print("=================================================================")
# print(hist.history)     # loss 와 val_loss의 key, value를 합쳐놓은 것
# print("=================================================================")
# print(hist.history['loss'])
# print("=================================================================")
# print(hist.history['val_loss'])


# 그래프로 비교
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
# loss :  12.729743003845215
# RMSE : 3.5678786869767625
# R2 :  0.9976330443981015
#==================================================================================#

#================================ 적용 후 데이터 ===================================#
# loss : 25.755203247070312
# mse :  1323.28759765625
# RMSE : 36.37702154533174
# r2 스코어:  0.7539495137131228
#==================================================================================#

##================================ 느 낀 점 ========================================##
# epochs=5000으로 돌렸는데 earlyStopping으로 1397번째에서 멈춤 => overfit방지, 시간단축
# 그래프로 볼 수 있어서 조금은 이해가 됨
# 결과적으로 overfit과 activation 적용으로 성능이 향상된 것을 확인함
##=================================================================================##
