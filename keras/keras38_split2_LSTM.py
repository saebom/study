import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time


a = np.array(range(1, 101)) 
x_predict = np.array(range(96, 106))    # 100~106

size = 5    # x는 4개, y는 1개

def split_x(dataset, size):
    aaa = []
    ccc = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        if (i + size) < len(dataset):
            aaa.append(subset)
            ccc.append(dataset[i + size])        
        else:
            break
    return np.array(aaa)


bbb = split_x(a, size)
x = bbb[:, :-1]     
y = bbb[:,-1]    
print(bbb)
print(bbb.shape)    

ccc= split_x(x_predict, size)
y_predict = ccc[:, -1] 
print(ccc)
print(x, y, y_predict)  # y_predict = [102 103 104 105]
print(x.shape, y.shape, y_predict.shape)    # (95, 4) (95,) (5,)

# x의 shape = (행, 열, 몇 개씩 자르는지(timesteps)!!!) => timesteps
x = x.reshape(95, 4, 1)
print(x.shape) # (96, 4, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=32
)

# 모델 구성 및 평가 예측할 것!!!
#2. 모델구성
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, 
               input_shape=(4, 1), activation='relu'))    #  [batch, timesteps, feature]   
model.add(LSTM(100, return_sequences=False, activation='relu'))         
model.add(Dense(64, activation='relu'))         
model.add(Dense(32, activation='relu'))         
model.add(Dense(1))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
print(date)

filepath = './_ModelCheckPoint/k35/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '01_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x, y, epochs=256, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time



#4. 평가, 예측 
loss = model.evaluate(x, y)
result = model.predict(x_test)
print('loss : ', loss)
print('[102 103 104 105]의 결과 : ', result)

#=============================== 이전 결과값 ====================================#
# loss :  0.16355521976947784
# [102 103 104 105]의 결과 :  [[67.932106 ]
#================================================================================#
