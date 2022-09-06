import numpy as np
from keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
import tensorflow as tf
print(tf.__version__)


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.  
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.  

# from keras.utils.np_utils import to_categorical
# x_train = to_categorical(y_train)
# x_test =to_categorical(y_test)


#2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu'):
     
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='sparse_categorical_crossentropy')
    
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropout = [0.3, 0.4, 0.5]
    activation = ['relu', 'linear', 'sigmoid', 'selu', 'elu']
    return {'batch_size' : batchs, 'optimizer': optimizers,
            'drop': dropout, 'activation':activation}
    
hyperparamers = create_hyperparameter()
# print(hyperparamers)
# {'batch_size': [100, 200, 300, 400, 500], 'optimizer': ['adam', 'rmsprop', 'adadelta'],
# 'drop': [0.3, 0.4, 0.5], 'activation': ['relu', 'linear', 'sigmoid', 'selu', 'elu']}

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier   # model을 wrapping해 주어야 함
keras_model = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
model = RandomizedSearchCV(keras_model, hyperparamers, cv=2, n_iter=2, verbose=2)  # n_iter=10 디폴트

import time
start = time.time()
model.fit(x_train, y_train, epochs=3, validation_split=0.4)
end = time.time()

print('걸린시간 : ', end - start)
print('model.best_params_ : ', model.best_params_)
print('model.best_estimator_ :', model.best_estimator_)
print('model.best_score_ : ', model.best_score_)
print('model.score : ', model.score)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))

# 걸린시간 :  96.72995018959045
# model.best_params_ :  {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 200, 'activation': 'selu'}
# model.best_estimator_ : <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001D67D1279A0>
# model.best_score_ :  0.9068333268165588
# model.score :  <bound method BaseSearchCV.score of RandomizedSearchCV(cv=5,
#                    estimator=<keras.wrappers.scikit_learn.KerasClassifier object at 0x000001D67F0EBF70>,
#                    param_distributions={'activation': ['relu', 'linear',
#                                                        'sigmoid', 'selu',
#                                                        'elu'],
#                                         'batch_size': [100, 200, 300, 400, 500],
#                                         'drop': [0.3, 0.4, 0.5],
#                                         'optimizer': ['adam', 'rmsprop',
#                                                       'adadelta']},
#                    verbose=2)>
#
# accuracy_score :  0.9232

# **** 주의 : version 이슈 있음!!! (sprase_categorical_crossentropy와 argmax 안먹음)
#             참고 사이트 https://stackoverflow.com/questions/44806125/attributeerror-model-object-has-no-attribute-predict-classes
# 