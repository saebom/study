import numpy as np
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

# One Hot Encoding 
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=64))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.save("./_save/keras23_13_save_model1.h5")
# model.save_weights("./_save/keras23_13_save_weights1.h5")

# model = load_model("./_save/keras23_13_save_model1.h5")
# model.load_weights('./_save/keras23_13_save_weights1.h5')
model.load_weights('./_save/keras23_13_save_weights2.h5')


# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중분류에서 loss = 'categorical_crossentropy'를 사용함
              metrics=['accuracy'])   

# earlyStopping = EarlyStopping(monitor = 'val_loss', patience=50, mode='min', verbose=1, 
#                               restore_best_weights=True)
# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=10, 
#                  validation_split=0.2,
#                  callbacks=[earlyStopping],
#                  verbose=1)
# end_time = time.time() - start_time

# model.save("./_save/keras23_13_save_model2.h5")
# model.save_weights("./_save/keras23_13_save_weights2.h5")
# model = load_model("./_save/keras23_13_save_model2.h5")


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)       
print('accuracy : ', acc)



#===================== save_model1 (random 모델) =================================#
# loss :  0.09918909519910812
# accuracy :  0.9740740656852722
#=================================================================================#

#===================== save_model2 (훈련한 모델) ==================================#
# loss :  0.11847227811813354
# accuracy :  0.9759259223937988
#=================================================================================#

#===================== save_weights1 (random 한 가중치 값) ========================#
# loss :  2.5163450241088867
# accuracy :  0.08703703433275223
#=================================================================================#

#===================== save_weights2 (훈련한 가중치 값) ===========================#
# loss :  0.11847227811813354
# accuracy :  0.9759259223937988
#=================================================================================#

