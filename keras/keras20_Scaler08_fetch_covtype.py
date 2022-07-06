import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_covtype
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
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

# One Hot Encoding 
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
y = y.reshape(-1, 1)
onehot_encoder.fit(y)
y = onehot_encoder.transform(y)

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
model.add(Dense(100, activation='linear', input_dim=54))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중분류에서 loss = 'categorical_crossentropy'를 사용함
              metrics=['accuracy'])   

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=50, mode='min', verbose=1, 
                              restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=5000, batch_size=128, 
                 validation_split=0.2,
                #  validation_data=(x_val,y_val),
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)       
print('accuracy : ', acc)

y_predict = model.predict(x_test)
y_predict = y_predict.argmax(axis=1)      # tensorflow에서 사용 : to_categorical
y_test = y_test.argmax(axis=1)            # tensorflow에서 사용 : to_categorical
# y_predict = tf.argmax(y_predict, axis=1)    # pandas에서 사용 : get_dummies
# y_test = tf.argmax(y_test, axis=1)          # pandas에서 사용 : get_dummies



print("================================ y_predict =================================")
print(y_predict)
print(y_test)
print("============================================================================")

acc = accuracy_score(y_test, y_predict)
print("============================================================================")   
print('acc 스코어 : ', acc)  


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
# loss :  0.3247310519218445
# accuracy_score : 0.8665696978569031
#==================================================================================#

#=========================== MinMaxScaler 적용 후 데이터 ===========================#
# loss :  0.15861260890960693
# accuracy_score : 0.9407758858087021
#==================================================================================#

#=========================== StandardScaler 적용 후 데이터 =========================#
# loss :  0.1636001318693161
# accuracy :  0.9392957091331482
#==================================================================================#

#=========================== MaxAbsScaler 적용 후 데이터 =========================#
# loss :  0.16479140520095825
# accuracy :  0.9384007251698182
#==================================================================================#

#=========================== RobustScaler 적용 후 데이터 =========================#
# loss :  0.15209655463695526
# accuracy :  0.9426117587662933
#==================================================================================#