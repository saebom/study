import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time


a = np.array(range(1, 101)) 
x_predict = np.array(range(96, 106))    # 100~106
size1 = 5    # x는 4개, y는 1개
size2 = 7

def split_x(dataset, size):
    aaa = []
    ccc = []
    if size1:
        for i in range(len(dataset) - size + 1):   
            subset = dataset[i : (i + size)]         
            aaa.append(subset)
        return np.array(aaa)
    else: 
        for i in range(len(dataset) - size):
            subset = dataset[i : (i +size-1)]         
            ccc.append(subset)
        return np.array(ccc)
        
bbb = split_x(a, size1)
print(bbb)
print(bbb.shape)    

ccc= split_x(x_predict, size2)
print(ccc)

x = bbb[:, :-1]     
y = bbb[:,-1]    
y_predict = ccc[:, -1]  
 
print(x, y, y_predict)  # y_predict = [102 103 104 105]
print(x.shape, y.shape, y_predict.shape)    # (96, 4) (96,) (4,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=32
)


# 모델 구성 및 평가 예측할 것!!!
#2. 모델구성
model = Sequential()
model.add(Dense(units=100, input_shape=(4, )))    #  [batch, timesteps, feature]   
model.add(Dense(100, activation='relu'))         
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
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict)
result = model.predict(y_predict)
print('loss : ', loss)
print('[102 103 104 105]의 결과 : ', result)

