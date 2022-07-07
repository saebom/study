import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
# model = Sequential()
# model.add(Dense(100, activation='linear', input_dim=30))    # 기본 activation의 default 값은 'linear'
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))        # relu 는 히든레이어에서만 사용가능함
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1, activation = 'sigmoid'))     # output에서 activatoin = 'sigmoid' ==> 마지막 결과값이 0~1사이로 나옴
                                                # **** 이진 분류모델의 경우 반올림하여 0과 1로 결과값을 받음

# 함수형 모델
input1 = Input(shape=(30,))
dense1 = Dense(100)(input1)
dense2 = Dense(100, activation='relu')(dense1)
dense3 = Dense(100, activation='relu')(dense2)
dense4 = Dense(100, activation='relu')(dense3)
output1 = Dense(1, activation='sigmoid')(dense4)
model = Model(inputs=input1, outputs=output1)

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
            #   metrics=['accuracy'],
              metrics=['accuracy', 'mse'])  # 이진 분류함수의 경우 loss = 'binary_crossentropy' 이고 
                                            # 평가지표 metrics['accuracy']를 사용, 회귀모델의 경우 mse를 사용함

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True)
start_time = time.time() 
hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

#=== 과제 1. accuracy_score 완성하기 =========================================
y_predict = model.predict(x_test)

y_predict = y_predict.flatten()                 # 차원 펴주기
y_predict = np.where(y_predict > 0.5, 1 , 0)   # 0.5보다 크면 1, 작으면 0
print(y_predict)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
# ====================================================== 과제 1 끝 ===========

acc = accuracy_score(y_test, y_predict)
print("=====================================================================")   
print('acc 스코어 : ', acc)  

print("=================================================================")
print("걸린시간 : ", end_time)

# print("===================================================================")   
# print(hist)     
# print("===================================================================")
# print(hist.history)   
# print("=====================================================================")
# print(hist.history['loss'])
# print("=====================================================================")
# print(hist.history['val_loss'])


#그래프로 비교
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

#================================== Sequential 모델 ================================#
# 걸린시간 : 11.633490324020386
# loss :  0.20227737724781036
# mse :  0.025880448520183563
# accuracy:  0.9707602339181286
#==================================================================================#

#==================================== 함수형 모델 ==================================#
# 걸린시간 : 10.964984655380249
# loss :  0.27133965492248535
# mse :  0.023448213934898376
# accuracy:  0.9766081871345029
#==================================================================================#