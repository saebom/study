import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.svm import LinearSVC


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

# One Hot Encoding 
# from sklearn.preprocessing import OneHotEncoder
# onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
# y = y.reshape(-1, 1)
# onehot_encoder.fit(y)
# y = onehot_encoder.transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=72
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
# model = Sequential()
# model.add(Dense(100, activation='linear', input_dim=54))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(7, activation='softmax'))
model = LinearSVC()


#3. 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중분류에서 loss = 'categorical_crossentropy'를 사용함
#               metrics=['accuracy'])   

# earlyStopping = EarlyStopping(monitor = 'val_loss', patience=50, mode='min', verbose=1, 
#                               restore_best_weights=True)

# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=5000, batch_size=128, 
#                  validation_split=0.2,
#                 #  validation_data=(x_val,y_val),
#                  callbacks=[earlyStopping],
#                  verbose=1)
# end_time = time.time() - start_time
model.fit(x_train, y_train)


#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss)       
# print('accuracy : ', acc)
results = model.score(x_test, y_test)
print("결과 acc : ", results)

# y_predict = model.predict(x_test)
# y_predict = y_predict.argmax(axis=1)      # tensorflow에서 사용 : to_categorical
# y_test = y_test.argmax(axis=1)            # tensorflow에서 사용 : to_categorical
# y_predict = tf.argmax(y_predict, axis=1)    # pandas에서 사용 : get_dummies
# y_test = tf.argmax(y_test, axis=1)          # pandas에서 사용 : get_dummies



# print("================================ y_predict =================================")
# print(y_predict)
# print(y_test)
# print("============================================================================")

# acc = accuracy_score(y_test, y_predict)
# print("============================================================================")   
# print('acc 스코어 : ', acc)  


#================================ SVM 적용 결과 ===================================#
# 결과 acc :  0.7129956856985497
# =================================================================================
# loss :  0.15209655463695526
# accuracy :  0.9426117587662933
#==================================================================================#
