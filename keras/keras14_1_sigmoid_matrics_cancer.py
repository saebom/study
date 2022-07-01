import numpy as np
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
print(datasets.feature_names)

x = datasets['data']         # x = datasets.data 와 동일
y = datasets['target']       # y = datasets.target 와 동일
print(x.shape, y.shape)      # (569, 30) (569,)

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=0
)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=30))    # 기본 default 값은 activation = 'linear'
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))        # relu 는 히든레이어에서만 사용가능함
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))     # output에서 activatoin = 'sigmoid' ==> 마지막 결과치는 0~1사이로 나옴
                                                # **** 분류모델의 경우 반올림하여 0과 1로 결과값을 받음


#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
            #   metrics=['accuracy'],
              metrics=['accuracy', 'mse'])  # 이진분류 함수의 경우 loss = 'binary_crossentropy' 이고 
                                            # 평가지표 metrics['accuracy']를 사용, 회귀모델의 경우 mse를 사용함

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
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

#=== 과제 1. accuracy_score 완성하기 =========================================
y_predict = model.predict(x_test)

y_predict = y_predict.flatten() # 차원 펴주기
pred_class = np.where(y_predict > 0.5, 1 , 0) #0.5보다크면 2, 작으면 1
print(pred_class)

from sklearn.metrics import classification_report
y_predict = model.predict(x_test)
print(classification_report(y_test, pred_class))
# ====================================================== 과제 1 끝 ===========

acc = accuracy_score(y_test, pred_class)
# print("=================================================================")   
print('acc 스코어: ', acc)  

# print("=================================================================")   
# print(hist)     
# print("=================================================================")
# print(hist.history)   
print("=================================================================")
print(hist.history['loss'])
print("=================================================================")
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

#==================================================================================#
# loss :  0.18249185383319855
# accuracy :  0.9210526347160339
# mse : 0.05536344274878502   ==> 이진분류에서는 mse를 신뢰할 수 없음
# acc 스코어: 0.9210526315789473
#==================================================================================#

##=============================== 내 용 정 리 ======================================##
# 1) model 구성에서 마지막 output layer에 activation=sigmoid
#   (이진분류 모델의 경우 반올림하여 0과 1의 결과 값을 받음)
# 2) model.compile 에서 loss='binary_crossentropy', metrics=['accuracy']
#   (이진분류 모델은 loss가 binary_crossentropy이고 accuracy를 평가지표로 사용함)
##=================================================================================##
         
##================================ 느 낀 점 ========================================##
# 히든 레이어에 activation = 'relu' 적용 및 output 레이어에 activatoin = 'sigmoid' 
# 적용으로 loss값이 좋아졌으며, accuracy 값을 통해 검증할 수 있어서 좋음!!!
##=================================================================================##
         
