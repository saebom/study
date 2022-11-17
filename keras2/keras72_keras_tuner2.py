# 일반적으로 라이브러리의 버전이 ex)1.xx 라면 배포가 된 버전 상용화된 버전이다.
# 버전이 0.xx 라면 아직 프로토 타입 및 베타테스트다. 이건 개발자들의 암묵적 합의다.
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
# from keras.optimizer_v2 import adam # tf 2.7 
from keras.optimizers import Adam     # tf 2.8 이상
from sklearn.metrics import accuracy_score


(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train/255. ,x_test/255.

def get_model(hp):
    # Dense의 노드 
    hp_unit1 = hp.Int('units1',min_value = 16,max_value=512,step=16)    # Int 대문자, 클래스 형태, 범위 지정
    hp_unit2 = hp.Int('units2',min_value = 16,max_value=512,step=16)
    hp_unit3 = hp.Int('units3',min_value = 16,max_value=512,step=16)
    hp_unit4 = hp.Int('units4',min_value = 16,max_value=512,step=16)
    
    # activation 
    hp_act1 = hp.Choice('activation1',values=['relu','selu','elu'])
    hp_act2 = hp.Choice('activation2',values=['relu','selu','elu'])
    hp_act3 = hp.Choice('activation3',values=['relu','selu','elu'])
    hp_act4 = hp.Choice('activation4',values=['relu','selu','elu'])
    
    # dropout 비율 
    hp_drop1 = hp.Choice('dropout1',values =[0.0, 0.2, 0.3, 0.4, 0.5])
    hp_drop2 = hp.Choice('dropout2',values =[0.0, 0.2, 0.3, 0.4, 0.5])
    hp_drop3 = hp.Choice('dropout3',values =[0.0, 0.2, 0.3, 0.4, 0.5])
    hp_drop4 = hp.Choice('dropout4',values =[0.0, 0.2, 0.3, 0.4, 0.5])
    
    # lr의 수치 
    hp_lr = hp.Choice('learning_rate', values = [1e-2,5e-3,1e-3,5e-4,1e-4])
    
    hp_batch = hp.Choice('batch_size', values = [24, 48, 72, 96])
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(hp_unit1, activation=hp_act1))
    model.add(Dropout(hp_drop1))
    
    model.add(Dense(hp_unit2, activation=hp_act2))
    model.add(Dropout(hp_drop2))
    
    model.add(Dense(hp_unit3, activation=hp_act3))
    model.add(Dropout(hp_drop3))
    
    model.add(Dense(hp_unit4, activation=hp_act4))
    model.add(Dropout(hp_drop4))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=hp_lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model

kerastuner = kt.Hyperband(get_model,
                          directory = 'my_dir',
                          objective = 'val_acc',
                          max_epochs = 6,
                          project_name = 'kerastuner-mnist2')

kerastuner.search(x_train,y_train,
                  validation_data=(x_test,y_test),
                  epochs=5)
best_hps = kerastuner.get_best_hyperparameters(num_trials=2)[0]

print('best parameter - units1 : ',best_hps.get('units1'))
print('best parameter - units2 : ',best_hps.get('units2'))
print('best parameter - units3 : ',best_hps.get('units3'))
print('best parameter - units4 : ',best_hps.get('units4'))

print('best parameter - dropout1 : ',best_hps.get('dropout1'))
print('best parameter - dropout2 : ',best_hps.get('dropout2'))
print('best parameter - dropout3 : ',best_hps.get('dropout3'))
print('best parameter - dropout4 : ',best_hps.get('dropout4'))

print('best parameter - activation1 : ',best_hps.get('activation1'))
print('best parameter - activation2 : ',best_hps.get('activation2'))
print('best parameter - activation3 : ',best_hps.get('activation3'))
print('best parameter - activation4 : ',best_hps.get('activation4'))

print('best parameter - learning_rate : ',best_hps.get('learning_rate'))

from keras.callbacks import EarlyStopping,ReduceLROnPlateau
es = EarlyStopping(patience=10,mode='min',monitor='val_loss',verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5,mode='auto',verbose=1,
                              factor=0.5)
model = kerastuner.hypermodel.build(best_hps)

history = model.fit(x_train,y_train,
          validation_split=0.2,epochs=5,
        #   callbacks=[es,reduce_lr]
          )
loss, accuracy = model.evaluate(x_test,y_test)
print('TR_ACC : ',accuracy)
print('TR_LOSS : ',loss)
y_predict = model.predcit(x_test)

print('TEST_ACC :',accuracy_score(y_test,y_predict))




