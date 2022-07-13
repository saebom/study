import numpy as np
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense, SimpleRNN, GRU
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

import tensorflow as tf
tf.random.set_seed(66)

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=32
)

#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(10, return_sequences=True, input_shape=(3, 1)))    #  [batch, timesteps, feature]   (N, 3, 1) => (N,3,10)
model.add(LSTM(5, return_sequences=False))         
model.add(Dense(1))
model.summary()

#  [실습] LSTM 2개 엮은거 테스트해보고

