import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,7,11,9,7])

#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
# from tensorflow.python.keras.optimizer_v1 import Adam, Adadelta, Adagrad, Adamax
# from tensorflow.python.keras.optimizer_v1 import RMSprop, SGD, Nadam
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax
from tensorflow.python.keras.optimizer_v2 import rmsprop, nadam


# learning_rate = 0.0001
learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.5, 0.05, 0.005, 0.0005]
for lr in learning_rates:
      optimizer1 = adam.Adam(learning_rate=lr)
      optimizer2 = adadelta.Adadelta(learning_rate=lr)
      optimizer3 = adagrad.Adagrad(learning_rate=lr)
      optimizer4 = adamax.Adamax(learning_rate=lr)
      optimizer5 = rmsprop.RMSProp(learning_rate=lr)
      optimizer6 = nadam.Nadam(learning_rate=lr)

      optimizers = [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5, optimizer6]

      for optimizer in optimizers:
            # model.compile(loss='mse', optimizer='adam')
            model.compile(loss='mse', optimizer=optimizer)  # optimizer learning_rate=0.001 디폴트

            model.fit(x, y, epochs=50, batch_size=1, verbose=0)

            #4. 평가, 예측
            loss = model.evaluate(x, y)
            y_predict = model.predict([11])

            optimizer_name = optimizer.__class__.__name__
            print('loss : ', round(loss, 4), 'lr : ', lr, 
                  '{0} 결과물 : {1}'.format(optimizer_name, y_predict))
      
# loss :  4.8911 lr :  0.1 Adam 결과물 : [[7.7551303]]
# loss :  23.039 lr :  0.1 Adadelta 결과물 : [[2.7869766]]
# loss :  499796.3125 lr :  0.1 Adagrad 결과물 : [[1266.7533]]
# loss :  218427.8438 lr :  0.1 Adamax 결과물 : [[-648.5109]]
# loss :  1813706368.0 lr :  0.1 RMSprop 결과물 : [[50372.023]]
# loss :  36521.7578 lr :  0.1 Nadam 결과물 : [[-195.86046]]

# loss :  2499.7183 lr :  0.01 Adam 결과물 : [[-86.79783]]
# loss :  2.1926 lr :  0.01 Adadelta 결과물 : [[10.109143]]
# loss :  5.4213 lr :  0.01 Adagrad 결과물 : [[13.179838]]
# loss :  2.3736 lr :  0.01 Adamax 결과물 : [[9.697327]]
# loss :  4181542.5 lr :  0.01 RMSprop 결과물 : [[-1987.0826]]
# loss :  4.0403 lr :  0.01 Nadam 결과물 : [[13.173337]]

# loss :  12.6363 lr :  0.001 Adam 결과물 : [[9.438054]]
# loss :  2.1854 lr :  0.001 Adadelta 결과물 : [[10.1145115]]
# loss :  9.0686 lr :  0.001 Adagrad 결과물 : [[15.618685]]
# loss :  17.5479 lr :  0.001 Adamax 결과물 : [[3.938839]]
# loss :  373.8963 lr :  0.001 RMSprop 결과물 : [[-17.599424]]
# loss :  141.4914 lr :  0.001 Nadam 결과물 : [[30.21277]]

# loss :  4659.9736 lr :  0.0001 Adam 결과물 : [[131.65253]]
# loss :  520.3856 lr :  0.0001 Adadelta 결과물 : [[53.890522]]
# loss :  10.7868 lr :  0.0001 Adagrad 결과물 : [[15.213812]]
# loss :  7.6824 lr :  0.0001 Adamax 결과물 : [[13.607168]]
# loss :  34141.457 lr :  0.0001 RMSprop 결과물 : [[339.94098]]
# loss :  707.766 lr :  0.0001 Nadam 결과물 : [[61.525753]]

# loss :  17.6198 lr :  1e-05 Adam 결과물 : [[17.111734]]
# loss :  6.4002 lr :  1e-05 Adadelta 결과물 : [[13.931803]]
# loss :  2.2275 lr :  1e-05 Adagrad 결과물 : [[9.971109]]
# loss :  2.7351 lr :  1e-05 Adamax 결과물 : [[11.653243]]
# loss :  4.9925 lr :  1e-05 RMSprop 결과물 : [[6.863944]]
# loss :  3.1817 lr :  1e-05 Nadam 결과물 : [[8.325617]]

# loss :  12.1258 lr :  0.5 Adam 결과물 : [[7.3269224]]
# loss :  162313240576.0 lr :  0.5 Adadelta 결과물 : [[-643492.8]]
# loss :  47565987840.0 lr :  0.5 Adagrad 결과물 : [[-259501.48]]
# loss :  4771.5552 lr :  0.5 Adamax 결과물 : [[92.04263]]
# loss :  3.3459506350810726e+17 lr :  0.5 RMSprop 결과물 : [[-9.260922e+08]]
# loss :  993658.8125 lr :  0.5 Nadam 결과물 : [[-845.26605]]

# loss :  2.1944 lr :  0.05 Adam 결과물 : [[11.19384]]
# loss :  7100394.5 lr :  0.05 Adadelta 결과물 : [[5732.194]]
# loss :  6.3838 lr :  0.05 Adagrad 결과물 : [[6.834178]]
# loss :  2.1246 lr :  0.05 Adamax 결과물 : [[11.438379]]
# loss :  225770471424.0 lr :  0.05 RMSprop 결과물 : [[-900104.25]]
# loss :  5.043 lr :  0.05 Nadam 결과물 : [[13.421872]]

# loss :  692991616.0 lr :  0.005 Adam 결과물 : [[-50811.746]]
# loss :  3539445.25 lr :  0.005 Adadelta 결과물 : [[-2464.091]]
# loss :  9.8895 lr :  0.005 Adagrad 결과물 : [[15.800234]]
# loss :  4.3197 lr :  0.005 Adamax 결과물 : [[12.847078]]
# loss :  4222829312.0 lr :  0.005 RMSprop 결과물 : [[-110626.055]]
# loss :  34.1043 lr :  0.005 Nadam 결과물 : [[18.797281]]

# loss :  141422528.0 lr :  0.0005 Adam 결과물 : [[-9855.943]]
# loss :  29588860.0 lr :  0.0005 Adadelta 결과물 : [[-695.91095]]
# loss :  2.6123 lr :  0.0005 Adagrad 결과물 : [[10.251869]]
# loss :  2.2034 lr :  0.0005 Adamax 결과물 : [[10.13393]]
# loss :  17353670.0 lr :  0.0005 RMSprop 결과물 : [[-7612.2124]]
# loss :  94965760.0 lr :  0.0005 Nadam 결과물 : [[-11901.687]]