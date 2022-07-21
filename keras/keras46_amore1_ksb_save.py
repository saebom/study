import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import tensorflow as tf
print(tf.__version__)   # 2.8.2

## 시험문제 ##
# 삼성전자와 아모레 주가로 아모레 가격 맞추기
# 가격 데이터에서 컬럼 7개 이상씩 추출(그 중 거래량은 반드시 들어가게 함)
# timesteps와 feature는 알아서 자르고
# 제공된 데이터 외 추가 데이터 사용금지
# 아모레 19일 아침 시가 맞추기 (점수배점 0.3)
# 아모레 20일 종가 맞추기 (점수배점 0.7)

#1. 데이터
path = './_data/test_amore_0718/'
amore = pd.read_csv(path +"아모레220718.csv", thousands=",", encoding='cp949')  # 마이크로소프트 'cp949' / 맥과 리눅스는 'utf8'
samsung = pd.read_csv(path +"삼성전자220718.csv", thousands=",", encoding='cp949')  # 마이크로소프트 'cp949' / 맥과 리눅스는 'utf8'

print(amore.shape, samsung.shape)    # (3180, 17), (3040, 17)

print(amore.describe)
print(amore.head(5))
print(amore.isnull().sum())
print(amore.info())

print(samsung.describe)
print(samsung.head(5))
print(samsung.isnull().sum())
print(samsung.info())

#일자 변환
amore['일자'] = pd.to_datetime(amore['일자'], infer_datetime_format=True)
samsung['일자'] = pd.to_datetime(samsung['일자'], infer_datetime_format=True)

#행 맞춰주기
amore = amore.drop(range(878, 3180), axis=0)
samsung = samsung.drop(range(878, 3040), axis=0)

#과거에서 현재 순으로 정렬
amore = amore.loc[::-1].reset_index(drop=True)
samsung = samsung.loc[::-1].reset_index(drop=True)
print(amore.head(5))
print(samsung.head(5))

#str 타입 삭제
amore = amore.drop(columns=['전일비'], axis=1)
samsung = samsung.drop(columns=['전일비'], axis=1)

#dataframe으로 변환
amore = amore.apply(pd.to_numeric) # convert all columns of DataFrame
samsung = samsung.apply(pd.to_numeric) # convert all columns of DataFrame

print(amore.head(5))
print(amore.describe)
print(amore.shape)  # (878, 16)

print(samsung.head(5))
print(samsung.describe)
print(samsung.shape)  # (878, 16)

#dataset 만들기
dataset_a = amore.drop(columns=['일자', '금액(백만)', '신용비', '등락률', '외인(수량)', '프로그램', '외인비'], axis=1) 
dataset_a = dataset_a[['고가', '저가', '종가', 'Unnamed: 6', '거래량', '개인', '기관', '시가']]
dataset_a = np.array(dataset_a)
print(dataset_a)

dataset_s = samsung.drop(columns=['일자', '금액(백만)', '신용비', '등락률', '외인(수량)', '프로그램', '외인비'], axis=1) 
dataset_s = dataset_s[['고가', '저가', '종가', 'Unnamed: 6', '거래량', '개인', '기관', '시가']]
dataset_s = np.array(dataset_s)
print(dataset_s)

time_steps = 5
y_column = 3

def split_xy(dataset, time_steps, y_column):                 
    x= []
    y= []
    for i in range(len(dataset)):
        x_end_number = i + time_steps      # 0 + 5   > 5
        y_end_number = x_end_number + y_column     # 5 + 3 -1 > 7
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :-1]                   # 0 : 5 , : -1   > 0행~4행, 마지막열 뺀 전부
        tmp_y = dataset[x_end_number-1:y_end_number, -1]       # 5 - 1 : 7 , -1  > 마지막 열의 4~6행
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy(dataset_a, time_steps, y_column)
x2, y2 = split_xy(dataset_s, time_steps, y_column)
print(x1.shape, y1.shape, x2.shape, y2.shape) # (871, 5, 7) (871, 4) (871, 5, 7) (871, 4)


#train, test split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test \
= train_test_split(x1, x2, y1, y2, shuffle=False, train_size=0.8, random_state=0)

print(x1_train.shape, x1_test.shape, y1_train.shape, y1_test.shape) # (696, 5, 7) (175, 5, 7) (696, 4) (175, 4)
print(x2_train.shape, x2_test.shape, y2_train.shape, y2_test.shape) # (696, 5, 7) (175, 5, 7) (696, 4) (175, 4)


x1_train = x1_train.reshape(696, 5*7, 1) 
x1_test = x1_test.reshape(175, 5*7, 1)
print(x1_train.shape)    # (696, 35, 1)
print(np.unique(x1_train, return_counts=True))

x2_train = x2_train.reshape(696, 5*7, 1) 
x2_test = x2_test.reshape(175, 5*7, 1)
print(x2_train.shape)    # (696, 35, 1)
print(np.unique(x2_train, return_counts=True))


#2. 모델구성
#2-1. amore 모델
input1 = Input(shape=(5*7,1))
dense1 = LSTM(128, return_sequences=True, 
               activation='relu', name='sb1')(input1)
dense2 = LSTM(64, return_sequences=False, activation='relu', name='sb2')(dense1)
dense3 = Dense(32, activation='relu', name='sb3')(dense2)
output1 = Dense(1, activation='linear', name='out_sb1')(dense3)

#2-2. samsung 모델
input2 = Input(shape=(5*7,1))
dense11 = LSTM(128, return_sequences=True, 
               activation='relu', name='sb11')(input2)
dense12 = LSTM(64, return_sequences=False, activation='relu', name='sb12')(dense11)
dense13 = Dense(32, activation='relu', name='sb13')(dense12)
output2 = Dense(1, activation='linear', name='out_sb2')(dense13)

#2-3. concatenate
from tensorflow.python.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2], name='mg1')
# merge2 = Dense(32, activation='relu', name='mg2')(merge1)
# last_output1 = Dense(1, name='last1')(merge2)

#2-4. amore output
output21 = Dense(128, activation='relu', name='op1')(merge1)
output22 = Dense(64, activation='relu', name='op2')(output21)
last_output2 = Dense(1, name='last2')(output22)

#2-5. samsung output
output31 = Dense(128, activation='relu', name='op11')(merge1)
output32 = Dense(64, activation='relu', name='op12')(output31)
last_output3 = Dense(1, name='last3')(output32)


# 모델 정의
model = Model(inputs=[input1, input2], outputs=[last_output2, last_output3])
model.summary()


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")   
print(date)

filepath = './_ModelCheckPoint/k46/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '_', date, '_', filename])
                      )
start_time = time.time() 
history = model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=300, batch_size=32,
                    validation_split=0.2, callbacks=[earlyStopping],
                    verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
print('loss :', loss) 

y1_predict, y2_predict = model.predict([x1_test, x2_test])
print('삼성 예측값 : ', y1_predict[-1])
print('아모레 예측값 : ', y2_predict[-1])




# ======================================= loss 및 예측값 ===========================================
# loss : [126822896.0, 16853078.0, 109969752.0]
# 아모레 예측값 :  [134404.27]

# ==================================================================================================
# model.summary()
# Model: "model"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_1 (InputLayer)            [(None, 5, 1)]       0
# __________________________________________________________________________________________________
# input_2 (InputLayer)            [(None, 5, 1)]       0
# __________________________________________________________________________________________________
# sb1 (LSTM)                      (None, 5, 128)       66560       input_1[0][0]
# __________________________________________________________________________________________________
# sb11 (LSTM)                     (None, 5, 128)       66560       input_2[0][0]
# __________________________________________________________________________________________________
# sb2 (LSTM)                      (None, 64)           49408       sb1[0][0]
# __________________________________________________________________________________________________
# sb12 (LSTM)                     (None, 64)           49408       sb11[0][0]
# __________________________________________________________________________________________________
# sb3 (Dense)                     (None, 32)           2080        sb2[0][0]
# __________________________________________________________________________________________________
# sb13 (Dense)                    (None, 32)           2080        sb12[0][0]
# __________________________________________________________________________________________________
# out_sb1 (Dense)                 (None, 1)            33          sb3[0][0]
# __________________________________________________________________________________________________
# out_sb2 (Dense)                 (None, 1)            33          sb13[0][0]
# __________________________________________________________________________________________________
# mg1 (Concatenate)               (None, 2)            0           out_sb1[0][0]
#                                                                  out_sb2[0][0]
# __________________________________________________________________________________________________
# op1 (Dense)                     (None, 128)          384         mg1[0][0]
# __________________________________________________________________________________________________
# op11 (Dense)                    (None, 128)          384         mg1[0][0]
# __________________________________________________________________________________________________
# op2 (Dense)                     (None, 64)           8256        op1[0][0]
# __________________________________________________________________________________________________
# op12 (Dense)                    (None, 64)           8256        op11[0][0]
# __________________________________________________________________________________________________
# last2 (Dense)                   (None, 1)            65          op2[0][0]
# __________________________________________________________________________________________________
# last3 (Dense)                   (None, 1)            65          op12[0][0]
# ==================================================================================================
# Total params: 253,572
# Trainable params: 253,572