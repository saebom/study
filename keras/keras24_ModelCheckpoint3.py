import numpy as ny
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
import time


#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=66
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()


#3. 훈련
model.compile(loss='mae', optimizer='adam',
              metrics=['mse']) 
earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True)  
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath='./_ModelCheckPoint/keras24_ModelCheckPoint3.hdf5'
                      )

start_time =time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, 
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1) 
end_time =time.time() - start_time

model.save('./_save/keras24_3_save_model.h5')


#4. 평가, 예측
print("=============================== 1. 기본 출력 ================================")
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)


print("============================= 2. load_model 출력 =============================")
model2 = load_model('./_save/keras24_3_save_model.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss2 : ', loss2)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2)
print("r2 스코어 : ", r2)


print("=========================== 3. ModelCheckpoint 출력 ===========================")
model3 = load_model('./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')
loss3 = model3.evaluate(x_test, y_test)
print('loss3 : ', loss3)

y_predict3 = model3.predict(x_test)
r2 = r2_score(y_test, y_predict3)
print("r2 스코어 : ", r2)


#================================= 1. 기본 출력 ===================================#
# loss2 :  [1.9132076501846313, 7.559213161468506]
# r2 스코어 :  0.9095603132709018
#=================================================================================#

#============================== 2. load_model 출력 ===============================#
# loss2 :  [1.9132076501846313, 7.559213161468506]
# r2 스코어 :  0.9095603132709018
#=================================================================================#

#============================ 3. ModelCheckpoint 출력 ============================#
# loss2 :  [1.9132076501846313, 7.559213161468506]
# r2 스코어 :  0.9095603132709018
#=================================================================================#