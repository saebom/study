import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# One Hot Encoding 
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)  # (150, 3)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=4))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))   # 결과값 label이 3개이므로 output 노드의 갯수는 3이 됨 
                                            # softmax를 통해 제일 큰 값이 선택됨(softmax의 값은 전체 합계 1.0이 됨)
                                            # class: 1) Iris-Setosa  2) Iris-Versicolour  3) Iris-Virginica

#3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중분류에서 loss = 'categorical_crossentropy'를 사용함
              metrics=['accuracy'])   

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=50, mode='min', verbose=1, 
                              restore_best_weights=True)
start_time = time.time()

hist = model.fit(x_train, y_train, epochs=1000, batch_size=10, 
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)       
print('accuracy : ', acc)
# results = model.evaluate(x_test, y_test)
# print('loss : ', results[0])      
# print('accuracy : ', results[1])


# print("========================= y_test[:5] =============================")
# print(y_test[:5])
# print("======================== y_predict[:5] ===========================")
# y_predict = model.predict(x_test[:5])  
# print(y_predict)
# print("==================================================================")

# y_test = np.argmax(y_test, axis=1)              # y_predict = y_predict.argmax(axis=1) 와 동일  
# y_predict = np.argmax(y_predict, axis=1)        # y_test = y_test.argmax(axis=1) 와 동일
y_predict = model.predict(x_test)
y_predict = y_predict.argmax(axis=1)
y_test = y_test.argmax(axis=1)
print("========================== y_predict =============================")
print(y_predict)
print(y_test)
print("==================================================================")

acc = accuracy_score(y_test, y_predict)
print("==================================================================")   
print('acc 스코어 : ', acc)  

# print("================================================================")   
# print(hist)     
# print("================================================================")
# print(hist.history)     
print("==================================================================")
print(hist.history['loss'])
print("==================================================================")
# print(hist.history['val_loss'])


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
# plt.legend(loc='upper right')  
plt.legend()   
plt.show()

#============================ Scaler 적용 전 데이터 ===============================#
# loss :  0.023589281365275383
# accuracy_score : 1.0
#==================================================================================#

#=========================== MinMaxScaler 적용 후 데이터 ===========================#
# loss :  0.14657451212406158
# accuracy_score : 0.9333333333333333
#==================================================================================#

#=========================== StandardScaler 적용 후 데이터 =========================#
# loss :  0.3094032406806946
# accuracy_score : 0.9111111111111111
#==================================================================================#

#=========================== MaxAbsScaler 적용 후 데이터 =========================#
# loss :  0.07578250765800476
# accuracy_score : 0.9555555555555556
#==================================================================================#

#=========================== RobustScaler 적용 후 데이터 =========================#
# loss :  0.3032987415790558
# accuracy_score : 0.9333333333333333
#==================================================================================#