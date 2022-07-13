import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import rc


# 1. 데이터

path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

submission = pd.read_csv(path + 'gender_submission.csv')
print('train.shape, test.shape, submit.shape', 
      train_set.shape, test_set.shape, submission.shape)    # (891, 11) (418, 10) (418, 2)

# 데이터 전처리
train_set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Ticket, Cabin, Name 삭제
train_set = train_set.drop(['Ticket', 'Cabin'], axis=1)
test_set = test_set.drop(['Ticket', 'Cabin'], axis=1)

train_set = train_set.drop(['Name'], axis=1)
test_set = test_set.drop(['Name'], axis=1)

# Age NaN값 변환
train_set['Age'] = train_set['Age'].fillna(train_set.Age.dropna().mode()[0])
test_set['Age'] = test_set['Age'].fillna(train_set.Age.dropna().mode()[0])


# Embarked, Sex NaN값 및 Object => int 변환
train_set['Embarked'] = train_set['Embarked'].fillna(train_set.Embarked.dropna().mode()[0])
train_set['Embarked'] = train_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

test_set['Embarked'] = test_set['Embarked'].fillna(test_set.Embarked.dropna().mode()[0])
test_set['Embarked'] = test_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

train_set['Sex'] = train_set['Sex'].fillna(train_set.Sex.dropna().mode()[0])
train_set['Sex'] = train_set['Sex'].map({'male':0, 'female':1}).astype(int)

test_set['Sex'] = test_set['Sex'].fillna(test_set.Sex.dropna().mode()[0])
test_set['Sex'] = test_set['Sex'].map({'male':0, 'female':1}).astype(int)

print(train_set.shape, test_set.shape)  # (891, 8) (418, 7)
print(train_set.head(5))
print(test_set.head(5))
print(train_set.isnull().sum())  
print(test_set.isnull().sum())  

# x, y 데이터
x = train_set.drop(['Survived'], axis=1)
print(x)
print(x.columns)    # 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'

y = train_set['Survived']
print(y)
print(y.shape)  # (891,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=23
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (801, 7) (90, 7) (801,) (90,)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(801, 7, 1, 1) 
x_test = x_test.reshape(90, 7, 1, 1)
print(x_train.shape)    
print(np.unique(x_train, return_counts=True))


# 2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(1, 1), padding='same', 
                 activation='relu', input_shape=(7, 1, 1)))
model.add(Dropout(0.25))     
model.add(Conv2D(64, (1, 1), padding='same', activation='relu'))                
model.add(Dropout(0.25))     
model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
model.add(Dropout(0.4))     
model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))   
model.add(Dropout(0.25))                 
model.add(Conv2D(64, (1, 1), padding='same', activation='relu'))                
model.add(Dropout(0.2))   
model.add(Conv2D(32, (1, 1), padding='same', activation='relu'))                
model.add(Dropout(0.2))   

model.add(Flatten())   
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.summary()


# 3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy', 'mse'])  

import datetime
date = datetime.datetime.now()     
date = date.strftime("%m%d_%H%M")   
print(date)

filepath = './_ModelCheckPoint/k24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '12_', date, '_', filename])
                      )
start_time = time.time() 
hist = model.fit(x_train, y_train, epochs=500, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
y_predict = y_predict.flatten()                 
y_predict = np.where(y_predict > 0.55, 1 , 0)   
print(y_predict)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))

acc = accuracy_score(y_test, y_predict)
print("=====================================================================")   
print('acc 스코어 : ', acc)  


#================================= 1. 기본 출력 ===================================#
# loss :  [0.40853211283683777, 0.1260533481836319]
# acc 스코어 :  0.8555555555555555
#=================================================================================#
