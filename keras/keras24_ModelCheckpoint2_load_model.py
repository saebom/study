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

#2. 모델 구성
# model = Sequential()
# model.add(Dense(64, input_dim=13))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))
# model.summary()

# #3. 훈련
# model.compile(loss='mae', optimizer='adam',
#               metrics=['mse']) 
# earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
#                               restore_best_weights=True)  
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
#                       save_best_only=True, 
#                       filepath='./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5'
#                       )

# start_time =time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, 
#                  validation_split=0.2,
#                  callbacks=[earlyStopping, mcp],
#                  verbose=1) 
# end_time =time.time() - start_time

model = load_model('./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)




#===================== save_weights1 (random 한 가중치 값) ========================#
# loss :  [2.269113540649414, 12.495966911315918]
# r2 스코어 :  0.8504961965410931
#=================================================================================#

#===================== save_weights2 (훈련한 가중치 값) ===========================#
# loss :  [2.187103271484375, 8.849693298339844]
# r2 스코어 :  0.8941208013595958
#=================================================================================#

#================== ModelCheckpoint (훈련 중 최적의 가중치 값) =====================#
# loss :  [1.892048716545105, 6.382836818695068]
# r2 스코어 :  0.9236346869739303
#=================================================================================#

#======================== load_model(ModelCheckPoint) ============================#
# loss :  [1.892048716545105, 6.382836818695068]
# r2 스코어 :  0.9236346869739303
#=================================================================================#
