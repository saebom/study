import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# from tensorflow.python.keras.models import Sequential, Model, load_model
# from tensorflow.python.keras.layers import LSTM, Dense, Input, Dropout
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.svm import LinearSVC


#1. 데이터
datasets = load_wine()
x, y = datasets.data, datasets.target

print(x.shape)  # (178, 13)
# x = x.reshape(178, 13, 1)
# print(x.shape)  # (178, 13, 1)

# One Hot Encoding 
# from tensorflow.python.keras.utils.np_utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)  # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

#2. 모델 구성
# model = Sequential()
# model.add(LSTM(100, activation='linear', input_shape=(13,1)))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(3, activation='softmax'))
model = LinearSVC()


#3. 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중분류에서 loss = 'categorical_crossentropy'를 사용함
#               metrics=['accuracy'])   

# import datetime
# date = datetime.datetime.now()      
# date = date.strftime("%m%d_%H%M")   
# print(date)

# filepath = './_ModelCheckPoint/k39/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# earlyStopping = EarlyStopping(monitor = 'val_loss', patience=50, mode='min', verbose=1, 
#                               restore_best_weights=True)

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
#                       save_best_only=True, 
#                       filepath="".join([filepath, '06_', date, '_', filename])
#                       )

# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=128, 
#                  validation_split=0.2,
#                  callbacks=[earlyStopping],
#                  verbose=1)
# end_time = time.time() - start_time
model.fit(x_train, y_train)


#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss)       
# print('accuracy : ', acc)

result = model.score(x_test, y_test)
print('결과 acc : ', result)

#================================ SVM 적용 결과 ===================================#
# 결과 acc :  0.9259259259259259
# =================================================================================
# loss :  0.0272836834192276
# accuracy :  0.9814814925193787
#==================================================================================#
